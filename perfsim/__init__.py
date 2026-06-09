"""perfsim: a benchmark library for performative prediction.

The central object is the distribution map D: Theta -> Delta(Z). Maps live
in perfsim.maps; environments adapt them (or true simulators) into the
Simulator epoch loop.
"""

from perfsim.environments import MapEnvironment
from perfsim.history import History
from perfsim.maps import (
    AccessError,
    DensityMap,
    DistributionMap,
    GaussianShiftMap,
    LocationScaleMap,
    MapAccess,
    MixtureShiftMap,
    ModelView,
    StrategicLinearMap,
    TransformationMap,
    access_levels,
)
from perfsim.simulator import Simulator

__all__ = [
    "AccessError",
    "DensityMap",
    "DistributionMap",
    "GaussianShiftMap",
    "History",
    "LocationScaleMap",
    "MapAccess",
    "MapEnvironment",
    "MixtureShiftMap",
    "ModelView",
    "Simulator",
    "StrategicLinearMap",
    "TransformationMap",
    "access_levels",
]
