"""Shared base for recommender worlds: frozen-theta run, grad_* path, accessors.

Subclasses set their state dict + knobs in __init__ and implement
`_forward(scores, state) -> (data, next_state, info)`. `info` must include
"exposure", "choice", "engagement".
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld
from perfsim.scenarios.recommender.entities import Corpus, UserPopulation


def _validate_n_steps(n_steps: int) -> None:
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")


class RecommenderWorldBase(StatefulPopulationWorld):
    """Common machinery; state carries "interest" (N, D) first as the per-agent axis."""

    def __init__(
        self,
        corpus: Corpus,
        users: UserPopulation,
        initial_state: State,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if corpus.dim != users.dim:
            raise ValueError(f"corpus dim {corpus.dim} != user dim {users.dim}")
        super().__init__(initial_state, dtype=dtype)
        self._features = corpus.features.to(dtype).detach().clone()
        self._appeal = corpus.appeal.to(dtype).detach().clone()
        self._interest0 = users.interest0.to(dtype).detach().clone()
        self._community = users.community.detach().clone()
        self._k = corpus.n_items
        self._d = corpus.dim
        self._n_users = users.n_users
        self._item_idx = torch.arange(self._k)
        self._gen: torch.Generator | None = None
        self._scores: Tensor | None = None          # frozen theta(features) per epoch
        self._info: dict[str, Tensor] = {}            # detached, for probes
        self._grad_info: dict[str, Tensor] = {}       # grad-tracked, for PerfGD

    # ---- contract / accessors ----------------------------------------------
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
        super().reset(seed=seed)            # restores state -> initial snapshot
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._scores = None
        self._info = {}
        self._grad_info = {}

    # ---- transition (subclass) ---------------------------------------------
    @abstractmethod
    def _forward(self, scores: Tensor, state: State) -> tuple[Data, State, dict[str, Tensor]]:
        """Pure, differentiable one-step transition: (data, next_state, info)."""

    def _scores_from(self, model: Model, *, grad: bool) -> Tensor:
        if grad:
            return model(self._features).reshape(-1).to(self._dtype)
        with torch.no_grad():
            return model(self._features).reshape(-1).to(self._dtype)

    @staticmethod
    def _detach_state(state: State) -> State:
        return {k: v.detach() for k, v in state.items()}

    def _cache_info(self, info: dict[str, Tensor]) -> None:
        self._info = {k: v.detach().clone() for k, v in info.items()}

    # ---- non-grad path (base sample/step + run) ----------------------------
    def _step(self, model: Model) -> tuple[Data, State]:
        scores = self._scores if self._scores is not None else self._scores_from(model, grad=False)
        data, next_state, info = self._forward(scores, self._state)
        self._cache_info(info)
        return data, self._detach_state(next_state)

    def run(self, model: Model, n_steps: int) -> Data:
        """Query theta once, evolve the population n_steps, return final data."""
        _validate_n_steps(n_steps)
        self._scores = self._scores_from(model, grad=False)
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)     # base.step -> _step -> installs next_state
        self._scores = None
        assert final is not None
        return final

    # ---- differentiable path (FullyDifferentiable / PerfGD) ----------------
    def grad_sample(self, model: Model) -> Data:
        scores = self._scores_from(model, grad=True)
        data, _next_state, info = self._forward(scores, self._state)
        self._grad_info = info
        self._cache_info(info)
        return data

    def grad_step(self, model: Model) -> Data:
        scores = self._scores_from(model, grad=True)
        data, next_state, info = self._forward(scores, self._state)
        self._grad_info = info
        self._cache_info(info)
        self._state = self._detach_state(next_state)
        return data

    def grad_run(self, model: Model, n_steps: int) -> Data:
        _validate_n_steps(n_steps)
        scores = self._scores_from(model, grad=True)   # grad live, queried once
        state = self._state
        final: Data | None = None
        info: dict[str, Tensor] = {}
        for _ in range(n_steps):
            final, state, info = self._forward(scores, state)
        self._grad_info = info
        self._cache_info(info)
        self._state = self._detach_state(state)
        assert final is not None
        return final
