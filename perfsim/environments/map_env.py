"""MapEnvironment: runs a DistributionMap in the Simulator epoch loop.

Maps are pure mathematical objects; this adapter gives one an execution
harness (RNG ownership, peek vs advance, batch size) so it binds to the
existing Simulator and learners unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from perfsim.core.environment import StatelessDynamics
from perfsim.maps.base import DistributionMap
from perfsim.core.types import Data, DataSchema

if TYPE_CHECKING:
    from perfsim.core.model import Model

# Capability methods forwarded to the map when it has them, so protocol
# checks like ClosedFormFixedPoint hold per instance.
_DELEGATED = ("closed_form_fp",)


class MapEnvironment(StatelessDynamics):
    """Adapter: DistributionMap in, Environment contract out."""

    def __init__(self, map_: DistributionMap, *, batch_size: int = 256) -> None:
        super().__init__()
        if not isinstance(map_, DistributionMap):
            raise TypeError(
                f"map_ must be a DistributionMap; got {type(map_).__name__}"
            )
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int; got {batch_size!r}")
        self._map = map_
        self._batch_size = batch_size

    @property
    def map(self) -> DistributionMap:
        return self._map

    def access(self, level: str, *, generator: "torch.Generator | None" = None):
        """A tier-restricted handle onto the map for a learner at `level`."""
        from perfsim.maps.access import MapAccess

        return MapAccess(self._map, level, generator=generator)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def produces_schema(self) -> DataSchema:
        return self._map.produces_schema

    def _sample_batch(self, model: "Model", generator: torch.Generator) -> Data:
        return self._map.sample(model, self._batch_size, generator=generator)

    def __getattr__(self, name: str) -> object:
        if name in _DELEGATED:
            return getattr(self._map, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
