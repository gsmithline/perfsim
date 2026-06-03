
"""
Social deimma game simpel game .
"""

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import Data, DataSchema, SUPERVISED_SCHEMA
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld



class PublicGoodsDilemma(StatefulPopulationWorld):
    """
    Class that has stateful population and return pop state


    y: (N,) current population giving propensities
    x: (N, D) fixed features sotred but these do not mutate
    type: fixed type baseline b(x_i) in [0, 1]
    """

    def __init__(
    self,
    x: Tensor,
    type_baseline: Tensor,
    y0: Tensor | None = None,
    *,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.0,
    eta: float = 0.5,
    k_sample: int = 5,
    r_multiplier: float = 2.0,
    dtype: torch.dtype = torch.float32,
) -> None:
        self.x = x
        self.type_baseline = type_baseline
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.k_sample = k_sample
        self.r_multiplier = r_multiplier
        self._gen: torch.Generator | None = None
        self._theta_pred: Tensor | None = None
        y_init = type_baseline.clone() if y0 is None else y0.to(dtype=dtype)
        initial_state = {
            "y": y_init.to(dtype=dtype),
            "x": x.to(dtype=dtype),
            "type_baseline": type_baseline.to(dtype=dtype),
        }
        super().__init__(initial_state, dtype=dtype)

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    def run(self, model: Model, n_steps: int) -> Data:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        with torch.no_grad():
            self._theta_pred = model(self._state["x"]).reshape(-1).to(self._dtype)
        final: Data | None = None
        for _ in range(n_steps):
            final = self.step(model)   # base step() calls your _step()
        assert final is not None
        return final
    

    def reset(self, seed: int = 0) -> None:
        super().reset(seed=seed)         
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._theta_pred = None
    
    def _step(self, model: Model) -> tuple[Data, State]: 
        y = self._state["y"]
        x = self._state["x"]

        b = self._state["type_baseline"]
        n= y.shape[0]

        pool = self.r_multiplier * y.mean()
        payoff = (1.0 - y) + pool

        idx = torch.randint(n, (n, self.k_sample), generator=self._gen)
        peer_pay = payoff[idx] 
        best = peer_pay.argmax(dim=1)

        j = idx[torch.arange(n), best]
        imitate = y + self.eta * (payoff[j] - payoff) * (y[j] - y)

        theta_pred = self._theta_pred
        if theta_pred is None:
            with torch.no_grad():
                theta_pred = model(self._state["x"]).reshape(-1).to(self._dtype)
        inertia = 1.0 - self.alpha - self.beta - self.gamma
        y_new = inertia * y + self.alpha * imitate + self.beta * b + self.gamma * theta_pred
        y_new = y_new.clamp(0.0, 1.0)

        data = {"x": x.clone(), "y": y_new.clone()}
        next_state = {"y": y_new, "x": x, "type_baseline": b}
        return data, next_state















        

