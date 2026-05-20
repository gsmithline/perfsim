"""Loss ABC.

Loss is a first-class object that takes (model, data) and returns a tensor.
This decouples "what we measure" from "how we optimize": metrics (PR, DPR)
use Loss objects to evaluate models on arbitrary (model, data) pairs without
going through a Learner.

Reduction is a keyword argument on `__call__`: "mean" (default), "sum", or
"none" (per-example).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from perfsim.core.model import Model
    from perfsim.core.types import Data


class Loss(ABC):
    """Loss function as a first-class object."""

    @abstractmethod
    def __call__(
        self, model: "Model", data: "Data", *, reduction: str = "mean"
    ) -> Tensor:
        """Compute loss on (model, data).

        Args:
            model: predictor.
            data: data dict containing at least the fields this Loss requires
                (typically `x`, `y` for supervised; see DataSchema).
            reduction: "mean", "sum", or "none" (per-example tensor).
        """
