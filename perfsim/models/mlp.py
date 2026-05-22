"""MLPModel: multi-layer perceptron predictor.

Outputs logits (no final activation by default). Pair with
`BCEWithLogitsLoss` for binary classification or with `MSELoss` for
regression. For probability outputs use `final_activation="sigmoid"`.

Initialization uses Xavier-uniform on weights and zero on biases. Zero
init (the convention for `LinearModel` in this codebase) is *not*
appropriate for MLPs: every hidden unit would compute the same function
and stay symmetric for the entire training run.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from perfsim.core.model import Model

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


class MLPModel(Model):
    """Feedforward MLP: 
    f(x) = WL σ(σ(W_1 x + b_1)) + bL. etc. 
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        out_features: int = 1,
        *,
        activation: str = "relu",
        final_activation: str | None = None,
        bias: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}"
                f"expected one of {sorted(_ACTIVATIONS)}"
            )
        if final_activation is not None and final_activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown final_activation {final_activation!r}; "
                f"expected one of {sorted(_ACTIVATIONS)} or None"
            )
        self._in_features = int(in_features)
        self._hidden_dims = tuple(int(h) for h in hidden_dims)
        self._out_features = int(out_features)
        self._activation_name = activation
        self._final_activation_name = final_activation

        dims = (self._in_features, *self._hidden_dims, self._out_features)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(
                nn.Linear(dims[i], dims[i + 1], bias=bias, device=device, dtype=dtype)
            )
            if i < len(dims) - 2:  # not the last linear
                layers.append(_ACTIVATIONS[activation]())
        if final_activation is not None:
            layers.append(_ACTIVATIONS[final_activation]())
        self.net = nn.Sequential(*layers)

        self._reset_parameters(init_seed)

    def _reset_parameters(self, seed: int | None) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    if generator is None:
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                            module.weight
                        )
                        bound = (6.0 / (fan_in + fan_out)) ** 0.5 
                        module.weight.uniform_(-bound, bound, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims

    @property
    def out_features(self) -> int:
        return self._out_features
