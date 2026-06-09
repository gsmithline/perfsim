"""Tier-restricted handle onto a map, the learner-side counterpart of
ModelView. A learner declaring a level may only call the capabilities at or
below it; anything higher raises AccessError. This is what makes "this method
used only level-T access" enforced rather than assumed.
"""

from __future__ import annotations

import torch

from perfsim.core.types import Data
from perfsim.maps.base import AccessError, DistributionMap, access_levels


_ORDER = {"samples": 1, "mechanism": 2, "density": 3}
_ALIASES = {"1b": "samples", "2a": "mechanism", "2b": "density"}


class MapAccess:
    def __init__(self, map_: DistributionMap, level: str,
                 *, generator: torch.Generator | None = None) -> None:
        level = _ALIASES.get(level, level)
        if level not in _ORDER:
            raise ValueError(
                f"level must be one of {tuple(_ORDER)} or {tuple(_ALIASES)}; got {level!r}"
            )
        if level not in access_levels(map_):
            raise AccessError(
                f"map exposes {sorted(access_levels(map_))}, cannot grant level {level}"
            )
        self._map = map_
        self._level = level
        self._gen = generator

    @property
    def level(self) -> str:
        return self._level

    def sample(self, model, n: int, *, generator: torch.Generator | None = None) -> Data:
        return self._map.sample(model, n, generator=self._pick(generator))

    def sample_base(self, n: int, *, generator: torch.Generator | None = None) -> Data:
        self._require("mechanism")
        return self._map.sample_base(n, generator=self._pick(generator))

    def transform(self, z_base: Data, model) -> Data:
        self._require("mechanism")
        return self._map.transform(z_base, self._map.view(model))

    def log_prob(self, z: Data, model) -> torch.Tensor:
        self._require("density")
        return self._map.log_prob(z, model)

    def _require(self, need: str) -> None:
        if _ORDER[self._level] < _ORDER[need]:
            raise AccessError(
                f"access level {self._level} may not use level-{need} capabilities"
            )

    def _pick(self, generator: torch.Generator | None) -> torch.Generator:
        g = generator or self._gen
        if g is None:
            raise ValueError("no generator: pass one here or to MapAccess(...)")
        return g
