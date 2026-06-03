"""ProducerEcosystemWorld: recommender ecosystem with strategic content supply.

Extends the demand-side loop with a producer population. Each of the K items
is a producer (a topic) carrying a *replicating* `availability` mass on the
simplex. The loop:

    theta scores -> exposure (optionally boosting small producers) ->
    users choose (affinity + exposure bias + supply weight) ->
    per-producer engagement -> replicator updates availability ->
    content mix shifts -> theta retrains.

This is the multi-agent ecosystem story of RecSim NG / Mladenov et al. 2020: a
myopic ranker drives rich-get-richer dynamics that collapse topic diversity and
lower user welfare; an ecosystem-aware `boost` can counteract it.

NOTE ON FIDELITY: the content dynamics here are a *replicator* over producer
availability share (Taylor-Jonker, as in perfsim's ReplicatorWorld), NOT RecSim
NG's own provider model (which accumulates unnormalized discounted engagement
and scales item production by it). The qualitative phenomenon -- rich-get-richer,
diversity collapse, welfare loss, boost helps -- matches; the mechanism differs.
This is a perfsim tool *inspired by* RecSim NG, not a port.

State = {"interest": (N, D), "availability": (K,)}. Differentiable end-to-end
(theta -> exposure -> choice -> engagement), so it satisfies FullyDifferentiable.

The producer channel is only performative when `gamma_supply > 0`: that is the
only path by which `availability` re-enters the observable engagement labels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld
from perfsim.scenarios.recommender.behaviors import (
    drift_interest,
    engagement,
    exposure_with_boost,
    producer_fitness,
    replicator_step,
)
from perfsim.scenarios.recommender.choice import affinities, mnl_choice
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


class ProducerEcosystemWorld(StatefulPopulationWorld):
    """Recommender ecosystem with a replicating content-producer population."""

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
        if corpus.dim != users.dim:
            raise ValueError(f"corpus dim {corpus.dim} != user dim {users.dim}")
        k = corpus.n_items
        availability0 = torch.full((k,), 1.0 / k, dtype=dtype)
        super().__init__(
            {"interest": users.interest0.to(dtype).clone(), "availability": availability0},
            dtype=dtype,
        )
        self._features = corpus.features.to(dtype).detach().clone()
        self._appeal = corpus.appeal.to(dtype).detach().clone()
        self._interest0 = users.interest0.to(dtype).detach().clone()
        self._community = users.community.detach().clone()
        self._k = k
        self._d = corpus.dim
        self._n_users = users.n_users
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp = float(temp)
        self.eta = float(eta)
        self.gamma_supply = float(gamma_supply)
        self.kappa = float(kappa)
        self.repl_rate = float(repl_rate)
        self.boost = float(boost)
        self.noise = float(noise)
        self._item_idx = torch.arange(self._k)
        self._gen: torch.Generator | None = None
        self._scores: Tensor | None = None
        self._info: dict[str, Tensor] = {}
        self._grad_info: dict[str, Tensor] = {}

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
    def last_availability(self) -> Tensor:
        return self._info["availability"]

    @property
    def grad_engagement(self) -> Tensor:
        return self._grad_info["engagement"]

    def hidden_quality(self) -> Tensor:
        """True item relevance implied by the ORIGINAL population (never trained on)."""
        with torch.no_grad():
            aff0 = self._interest0 @ self._features.t()
            return aff0.mean(dim=0) * self._appeal

    def reset(self, seed: int = 0) -> None:
        super().reset(seed=seed)   # restores interest -> interest0, availability -> uniform
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._scores = None
        self._info = {}
        self._grad_info = {}

    # ---- core transition ---------------------------------------------------
    def _forward(
        self, scores: Tensor, interest: Tensor, availability: Tensor
    ) -> tuple[Data, Tensor, Tensor, dict[str, Tensor]]:
        u = exposure_with_boost(scores, availability, beta=self.beta, boost=self.boost)
        log_u = torch.log(u.clamp_min(1e-12))
        log_a = torch.log(availability.clamp_min(1e-12))
        aff = affinities(interest, self._features)
        choice = mnl_choice(
            aff, log_u, alpha=self.alpha, temp=self.temp,
            log_availability=log_a, gamma_supply=self.gamma_supply,
        )
        y = engagement(choice, self._appeal)                 # (K,)
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
        return data, interest_next, availability_next, info

    def _step(self, model: Model) -> tuple[Data, State]:
        scores = self._scores
        if scores is None:
            with torch.no_grad():
                scores = model(self._features).reshape(-1).to(self._dtype)
        data, i_next, a_next, info = self._forward(
            scores, self._state["interest"], self._state["availability"]
        )
        self._info = {k: v.detach().clone() for k, v in info.items()}
        return data, {"interest": i_next.detach(), "availability": a_next.detach()}

    def run(self, model: Model, n_steps: int) -> Data:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        with torch.no_grad():
            self._scores = model(self._features).reshape(-1).to(self._dtype)
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)
        self._scores = None
        assert final is not None
        return final

    # ---- differentiable path (FullyDifferentiable / PerfGD) ----------------
    def grad_sample(self, model: Model) -> Data:
        scores = model(self._features).reshape(-1).to(self._dtype)
        data, _i, _a, info = self._forward(
            scores, self._state["interest"], self._state["availability"]
        )
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        return data

    def grad_step(self, model: Model) -> Data:
        scores = model(self._features).reshape(-1).to(self._dtype)
        data, i_next, a_next, info = self._forward(
            scores, self._state["interest"], self._state["availability"]
        )
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        self._state = {"interest": i_next.detach(), "availability": a_next.detach()}
        return data

    def grad_run(self, model: Model, n_steps: int) -> Data:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        scores = model(self._features).reshape(-1).to(self._dtype)
        interest = self._state["interest"]
        availability = self._state["availability"]
        final: Data | None = None
        info: dict[str, Tensor] = {}
        for _ in range(n_steps):
            final, interest, availability, info = self._forward(scores, interest, availability)
        self._grad_info = info
        self._info = {k: v.detach().clone() for k, v in info.items()}
        self._state = {"interest": interest.detach(), "availability": availability.detach()}
        assert final is not None
        return final
