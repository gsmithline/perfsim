"""Model base: parametric predictor.

Subclasses nn.Module to leverage torch parameter management; adds PP-flavored
methods get_params / set_params (flat tensor view of all parameters) and
clone (deep copy). Concrete subclasses define the architecture and override
forward.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    """Base parametric predictor.

    Subclasses must override `forward`. Inherits torch's `.parameters()` and
    `.state_dict()` for free; the PP-flavored API adds:

    - `get_params() -> Tensor`: flat view of all parameters, concatenated in
      `self.parameters()` order. Used by Worlds to read theta and by metrics
      to record trajectories.
    - `set_params(theta: Tensor)`: distribute a flat tensor back to the
      parameters. Used by Simulator off-policy evaluation (set a clone's
      params to a different theta).
    - `clone() -> Model`: deep copy. Independent params; mutating the clone
      does not affect the original. Used for decoupled-risk evaluation.
    """

    def get_params(self) -> Tensor:
        """Flat tensor view of all parameters, concatenated in
        `self.parameters()` order.
        """
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def set_params(self, theta: Tensor) -> None:
        """Set parameters from a flat tensor in the order produced by
        `get_params`.
        """
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                chunk = theta[offset : offset + n]
                p.copy_(chunk.reshape(p.shape).to(p.device, p.dtype))
                offset += n
        if offset != theta.numel():
            raise ValueError(
                f"theta has {theta.numel()} elements but model has {offset} params"
            )

    def clone(self) -> "Model":
        """Independent deep copy of the model."""
        return copy.deepcopy(self)

    @property
    def device(self) -> torch.device:
        """Device of the first parameter. Assumes all parameters share a device."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
