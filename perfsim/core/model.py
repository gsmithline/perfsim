"""Model base: parametric predictor with a flat-tensor param API for PP."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    """Base parametric predictor. Subclasses override `forward`.

    Adds a flat-tensor view over parameters (get_params/set_params) and clone,
    used by environments to read theta and by off-policy risk evaluation.
    """

    def get_params(self) -> Tensor:
        """Flat view of all parameters, concatenated in `self.parameters()` order."""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def set_params(self, theta: Tensor) -> None:
        """Set parameters from a flat tensor in `get_params` order."""
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
        """Device of the first parameter (assumes all share a device)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
