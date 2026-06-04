"""Stateful combinators: lift a stateless DistributionMap into stateful PP.

Transition maps Tr(theta, Q_{t-1}) (survey eqs 21-22) modeled empirically: the
state is a sample buffer, each step mixes the buffer with fresh D(theta) draws.
Canonical use is epoch_size=1 with the env persisting across rounds, so round t
applies one Tr(theta_t, Q_{t-1}).
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from perfsim.core.environment import StatefulDynamics
from perfsim.core.types import Data, DataSchema
from perfsim.maps.base import DistributionMap, validate_n

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perfsim.core.model import Model


def _resample(data: Data, n: int, generator: torch.Generator) -> Data:
    b = next(iter(data.values())).shape[0]
    idx = torch.randint(b, (n,), generator=generator)
    return {k: v[idx] for k, v in data.items()}


def _concat(a: Data, b: Data) -> Data:
    return {k: torch.cat([a[k], b[k]], dim=0) for k in a}


class _TransitionEnv(StatefulDynamics):
    """Shared harness: owns a map, batch size, RNG; sample peeks, step advances."""

    def __init__(self, map_: DistributionMap, *, batch_size: int = 256) -> None:
        if not isinstance(map_, DistributionMap):
            raise TypeError(f"map_ must be a DistributionMap; got {type(map_).__name__}")
        validate_n(batch_size)
        self._map = map_
        self._batch_size = int(batch_size)
        self._gen: torch.Generator | None = None

    @property
    def map(self) -> DistributionMap:
        return self._map

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def produces_schema(self) -> DataSchema:
        return self._map.produces_schema

    def reset(self, seed: int = 0) -> None:
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._init_state()

    def _draw(self, model: "Model") -> Data:
        if self._gen is None:
            self.reset(seed=0)
        return self._map.sample(model, self._batch_size, generator=self._gen)

    def step(self, model: "Model") -> Data:
        return self._transition(model, advance=True)

    def sample(self, model: "Model") -> Data:
        if self._gen is None:
            self.reset(seed=0)
        assert self._gen is not None
        saved = self._gen.get_state()
        out = self._transition(model, advance=False)
        self._gen.set_state(saved)
        return out

    @abstractmethod
    def _init_state(self) -> None: ...

    @abstractmethod
    def _transition(self, model: "Model", *, advance: bool) -> Data: ...


class GeometricDecayEnv(_TransitionEnv):
    """Tr(theta, Q) = lam * Q + (1 - lam) * D(theta), empirically (survey eq 21)."""

    def __init__(self, map_: DistributionMap, *, lam: float, batch_size: int = 256) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1]; got {lam}")
        super().__init__(map_, batch_size=batch_size)
        self._lam = float(lam)
        self._buffer: Data | None = None

    def _init_state(self) -> None:
        self._buffer = None

    def _transition(self, model: "Model", *, advance: bool) -> Data:
        fresh = self._draw(model)
        if self._buffer is None:
            out = fresh
        else:
            n_keep = int(round(self._lam * self._batch_size))
            keep = _resample(self._buffer, n_keep, self._gen)
            new = _resample(fresh, self._batch_size - n_keep, self._gen)
            out = _concat(keep, new)
        if advance:
            self._buffer = out
        return out


class StaggeredResponseEnv(_TransitionEnv):
    """Uniform mixture of the last k deployments' responses (survey eq 22)."""

    def __init__(self, map_: DistributionMap, *, k: int, batch_size: int = 256) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive int; got {k!r}")
        super().__init__(map_, batch_size=batch_size)
        self._k = k
        self._buffers: list[Data] = []

    def _init_state(self) -> None:
        self._buffers = []

    def _transition(self, model: "Model", *, advance: bool) -> Data:
        buffers = (self._buffers + [self._draw(model)])[-self._k :]
        pool = buffers[0]
        for b in buffers[1:]:
            pool = _concat(pool, b)
        out = _resample(pool, self._batch_size, self._gen)
        if advance:
            self._buffers = buffers
        return out
