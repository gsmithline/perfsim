"""Distribution maps: the central object of performative prediction.

base.py defines the ABCs and access machinery; sibling modules hold the
canonical families from the literature. This package depends only on
perfsim.core primitives (types, model), never on environments.
"""

from perfsim.maps.base import (
    AccessError,
    DensityMap,
    DistributionMap,
    ModelView,
    TransformationMap,
    access_levels,
)
from perfsim.maps.gaussian_shift import GaussianShiftMap
from perfsim.maps.location_scale import LocationScaleMap
from perfsim.maps.strategic import StrategicLinearMap

__all__ = [
    "AccessError",
    "DensityMap",
    "DistributionMap",
    "GaussianShiftMap",
    "LocationScaleMap",
    "ModelView",
    "StrategicLinearMap",
    "TransformationMap",
    "access_levels",
]
