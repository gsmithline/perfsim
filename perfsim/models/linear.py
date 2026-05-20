"""Linear and logistic regression Models.

`LinearModel`: f(x) = x W + b. Suitable for regression with MSE, or as logits
with cross-entropy / BCEWithLogits.

`LogisticModel`: f(x) = sigmoid(x W + b). Returns probabilities in (0, 1);
pair with BCELoss, not BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from perfsim.core.model import Model


class LinearModel(Model):
    """Linear predictor: f(x) = x W^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        bias: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.linear.weight.zero_()
            if bias:
                self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LogisticModel(LinearModel):
    """Logistic predictor: f(x) = sigmoid(x W^T + b). Outputs in (0, 1).

    Pair with `BCELoss`. For numerical stability prefer `LinearModel` +
    `BCEWithLogitsLoss` instead.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(x))
