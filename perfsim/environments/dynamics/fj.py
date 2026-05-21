"""FJWorld: linear Friedkin-Johnsen opinion dynamics on a graph.

Direct torch port of the inner loop from
`Opinion-dynamics-post-training/run_free_fj.py` (lines 94-101) and
`run_cf_fj.py` (lines 372-375). Treated here as a *non-ABM* macroscopic
setting: opinions are a per-agent scalar (or D-dim) vector updated by
matrix dynamics, not a per-agent decision rule. Faithfulness to the
source code is the goal; idiomatic perfsim refactor is intentionally
minimal.

Per PP round t:

    predictions = model(features)                           # (N,) or (N, D)
    x_zero      = (1 - platform_sus) * innate
                  + platform_sus * predictions
    x           = x_state   (the current opinion vector, persisted)
    for _ in range(n_ticks):
        x = peer_sus * x_zero + (1 - peer_sus) * (W @ x)

    data = {"x": features, "y": x_final}
    state["opinion"] = x_final                              # persists

`peer_sus` is per-agent (length-N tensor). The "platform" coupling
mirrors `run_cf_fj.py:372` exactly: the deployed predictor's per-agent
prediction blends with each agent's innate opinion to form the per-round
anchor `x_zero`. With `platform_sus = 0` the platform is absent and
`x_zero = innate` (matches `run_free_fj.py`).

Non-ABM in the sense that there are no per-agent decision rules. Every
agent applies the same fixed-coefficient linear update; the only
heterogeneity is per-agent peer_sus and innate opinion. This is what the
existing pokec_simulations / run_*_fj scripts treat as the canonical
"FJ" baseline.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld


def normalize_adjacency(adj: Tensor) -> Tensor:
    """Column-normalize an adjacency matrix the way run_free_fj.py does:

        degs_inv = 1 / col_sum
        W_norm   = adj * degs_inv[:, None]

    Matches numpy lines 86-88 in run_free_fj.py. Infinite / oversized inverse
    degrees (isolated nodes) are zeroed.
    """
    degs = adj.sum(dim=0)
    degs_inv = torch.where(degs > 0, 1.0 / degs, torch.zeros_like(degs))
    degs_inv = torch.where(degs_inv > 1.1, torch.zeros_like(degs_inv), degs_inv)
    return adj * degs_inv.unsqueeze(0)


class FJWorld(StatefulPopulationWorld):
    """Linear FJ on a graph with optional platform coupling.

    Args:
        innate:       (N,) or (N, D) per-agent innate opinion.
        graph:        (N, N) row- or column-normalized influence matrix.
                      Use ``normalize_adjacency`` if you have a raw adjacency.
        peer_sus:     (N,) per-agent peer susceptibility in [0, 1]. A value of
                      ``1`` means the agent does not mix in neighbor opinions
                      (fully stubborn at x_zero); ``0`` means the agent's
                      opinion is the neighbor-weighted average of its peers.
        platform_sus: scalar in [0, 1]. Mixes platform prediction into the
                      per-agent anchor x_zero. 0 ⇒ platform-free FJ.
        n_ticks:      inner-loop iterations per PP round (FJ_K in the source
                      scripts; they use 100).
        features:     optional (N, F) feature matrix passed to the predictor.
                      If None, uses ``innate`` as the feature (matches the
                      "predictor reads innate opinion" baseline).
    """

    def __init__(
        self,
        innate: Tensor,
        graph: Tensor,
        peer_sus: Tensor,
        *,
        platform_sus: float = 0.0,
        n_ticks: int = 100,
        features: Tensor | None = None,
        initial_opinion: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if innate.ndim not in (1, 2):
            raise ValueError(f"innate must be 1-D or 2-D; got shape {tuple(innate.shape)}")
        n = innate.shape[0]
        if graph.shape != (n, n):
            raise ValueError(f"graph must be ({n}, {n}); got {tuple(graph.shape)}")
        if peer_sus.shape != (n,):
            raise ValueError(f"peer_sus must be (N,); got {tuple(peer_sus.shape)}")
        if not (0.0 <= platform_sus <= 1.0):
            raise ValueError(f"platform_sus must be in [0, 1]; got {platform_sus}")
        if n_ticks < 1:
            raise ValueError(f"n_ticks must be >= 1; got {n_ticks}")

        innate_t = innate.to(dtype=dtype).detach().clone()
        x_init = innate_t.clone() if initial_opinion is None else initial_opinion.to(dtype=dtype).detach().clone()
        if x_init.shape != innate_t.shape:
            raise ValueError(
                f"initial_opinion shape {tuple(x_init.shape)} must match innate "
                f"shape {tuple(innate_t.shape)}"
            )

        super().__init__({"opinion": x_init}, dtype=dtype)
        self._innate = innate_t
        self._W = graph.to(dtype=dtype).detach().clone()
        # peer_sus enters as broadcast against (N,) or (N, D); store as (N, 1) when D>1.
        ps = peer_sus.to(dtype=dtype).detach().clone()
        self._peer_sus = ps.unsqueeze(-1) if innate_t.ndim == 2 else ps
        self._platform_sus = float(platform_sus)
        self._n_ticks = int(n_ticks)
        self._features = features.to(dtype=dtype).detach().clone() if features is not None else innate_t
        if self._features.shape[0] != n:
            raise ValueError(
                f"features must have N rows ({n}); got {tuple(self._features.shape)}"
            )
        self._n = n
        self._is_scalar = innate_t.ndim == 1

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_agents(self) -> int:
        return self._n

    @property
    def innate(self) -> Tensor:
        return self._innate.clone()

    @property
    def graph(self) -> Tensor:
        return self._W.clone()

    @property
    def features(self) -> Tensor:
        return self._features.clone()

    @property
    def platform_sus(self) -> float:
        return self._platform_sus

    @property
    def n_ticks(self) -> int:
        return self._n_ticks

    def _predict(self, model: Model) -> Tensor:
        # nn.Linear etc. expect (B, F); if features are 1-D, add a singleton.
        feat = self._features
        if feat.ndim == 1:
            feat = feat.unsqueeze(-1)
        with torch.no_grad():
            r = model(feat)
        if r.shape[0] != self._n:
            raise ValueError(
                f"model output leading dim {r.shape[0]} does not match N={self._n}"
            )
        # Squeeze trailing singleton dim if predictor returns (N, 1) and innate is (N,).
        if self._is_scalar and r.ndim == 2 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        if r.shape != self._innate.shape:
            raise ValueError(
                f"model output shape {tuple(r.shape)} does not match innate "
                f"shape {tuple(self._innate.shape)}"
            )
        return r.detach().to(self._dtype)

    def _inner_fj(self, x_zero: Tensor, x_init: Tensor) -> Tensor:
        """Run n_ticks of the FJ update.

        Mirrors `x_temp = peer_sus * x_zero + (1 - peer_sus) * (W_norm @ x_temp)`
        from run_free_fj.py:98 (and equivalently run_cf_fj.py:375).
        """
        x = x_init
        one_minus = 1.0 - self._peer_sus
        for _ in range(self._n_ticks):
            graph_term = self._W @ x
            x = self._peer_sus * x_zero + one_minus * graph_term
        return x

    def _step(self, model: Model) -> tuple[Data, State]:
        predictions = self._predict(model)
        x_zero = (1.0 - self._platform_sus) * self._innate + self._platform_sus * predictions
        x_final = self._inner_fj(x_zero, self._state["opinion"])
        # Emit data with a (N, F) feature matrix and (N, 1) label tensor when
        # the underlying innate/opinion is scalar. nn.Linear-style learners
        # require 2-D inputs.
        x_data = self._features.unsqueeze(-1) if self._features.ndim == 1 else self._features
        y = x_final.unsqueeze(-1) if self._is_scalar else x_final
        return {"x": x_data, "y": y}, {"opinion": x_final}

    def fj_equilibrium(self, x_zero: Tensor | None = None) -> Tensor:
        """Closed-form FJ fixed point for diagnostic / gating use.

        Solves for x* satisfying ``x_i^* = α_i x_zero_i + (1 - α_i) Σ_j W_ij x_j^*``.
        In matrix form, ``(I - diag(1 - α) W) x* = diag(α) x_zero``. Default
        ``x_zero = innate`` (the platform-free anchor).
        """
        if x_zero is None:
            x_zero = self._innate
        ps = self._peer_sus.reshape(-1)  # (N,)
        eye = torch.eye(self._n, dtype=self._dtype)
        A = eye - (1.0 - ps).unsqueeze(-1) * self._W  # diag(1-α) applied row-wise
        if self._is_scalar:
            rhs = ps * x_zero  # (N,) * (N,) -> (N,)
            return torch.linalg.solve(A, rhs)
        # (N, D) case: each feature column solved with the same matrix A.
        rhs = ps.unsqueeze(-1) * x_zero  # (N, 1) * (N, D) -> (N, D)
        return torch.linalg.solve(A, rhs)
