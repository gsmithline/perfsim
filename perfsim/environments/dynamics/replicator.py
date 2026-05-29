"""ReplicatorWorld: discrete-time replicator dynamics on a K-strategy mixture.

Taylor-Jonker 1978 discrete replicator:
p_{t+1} = p_t * (1 + f(p_t)) / <p_t, 1 + f(p_t)>, where f is the per-strategy
fitness (depends on the mixture and, in PP, on the deployed predictor). Each PP
round runs n_ticks of the update; emits per-strategy fitness, persists the mixture.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld

FitnessFn = Callable[[Tensor, Model], Tensor]


class ReplicatorWorld(StatefulPopulationWorld):
    """Discrete replicator dynamics on a K-strategy mixture.

    Args:
        p0:       (K,) initial mixture in the simplex (>= 0, sums to 1).
        fitness:  callable (p, model) -> (K,) per-strategy fitness. For the
                  platform-free baseline, ignore the model argument.
        n_ticks:  inner replicator iterations per PP round.

    Emits {"x": one-hot strategy id, "y": fitness vector}; persists state["mixture"].
    """

    def __init__(
        self,
        p0: Tensor,
        fitness: FitnessFn,
        *,
        n_ticks: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if p0.ndim != 1:
            raise ValueError(f"p0 must be 1-D (K,); got shape {tuple(p0.shape)}")
        if not torch.isclose(p0.sum(), torch.tensor(1.0, dtype=p0.dtype), atol=1e-6):
            raise ValueError(f"p0 must sum to 1; got sum={p0.sum().item()}")
        if (p0 < 0).any():
            raise ValueError(f"p0 must be non-negative; got min={p0.min().item()}")
        if n_ticks < 1:
            raise ValueError(f"n_ticks must be >= 1; got {n_ticks}")

        p0_t = p0.to(dtype=dtype).detach().clone()
        super().__init__({"mixture": p0_t}, dtype=dtype)
        self._k = p0_t.numel()
        self._fitness_fn = fitness
        self._n_ticks = int(n_ticks)

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_strategies(self) -> int:
        return self._k

    @property
    def n_ticks(self) -> int:
        return self._n_ticks

    def _replicator_step(self, p: Tensor, f: Tensor) -> Tensor:
        """Single Taylor-Jonker update. f is the per-strategy fitness vector."""
        f_shifted = f + 1.0
        return (p * f_shifted) / (p @ f_shifted).clamp(min=1e-12)

    def _step(self, model: Model) -> tuple[Data, State]:
        p = self._state["mixture"]
        last_f = torch.zeros(self._k, dtype=self._dtype)
        for _ in range(self._n_ticks):
            f = self._fitness_fn(p, model).to(self._dtype)
            if f.shape != (self._k,):
                raise ValueError(
                    f"fitness must return shape ({self._k},); got {tuple(f.shape)}"
                )
            p = self._replicator_step(p, f)
            last_f = f
        # Data: one-hot per-strategy id as features, fitness as label.
        x = torch.eye(self._k, dtype=self._dtype)
        y = last_f.unsqueeze(-1)
        return {"x": x, "y": y}, {"mixture": p}
