"""FJWorld: linear Friedkin-Johnsen opinion dynamics on a graph.

Direct torch port of the inner loop from
`Opinion-dynamics-post-training/run_free_fj.py` (lines 94-101) and
`run_cf_fj.py` (lines 372-375). Treated here as a non-ABM macroscopic
setting: opinions are a per-agent scalar (or D-dim) vector updated by
matrix dynamics, not a per-agent decision rule.

One `env.step(model)` performs one FJ update:

    predictions = model(features)                           # (N,) or (N, D)
    x_zero      = (1 - platform_sus) * innate
                  + platform_sus * predictions
    x_new       = peer_sus * x_zero + (1 - peer_sus) * (W @ x_state)
    data        = {"x": features, "y": x_new}
    state["opinion"] = x_new                                # persists

The user controls how many FJ iterations happen per outer round via the
Simulator's `epoch_size` (or by manually looping `world.step(model)`).
With `epoch_size = K`, the Simulator drives K consecutive FJ updates
under a frozen deployed model before retraining; the opinion vector
evolves continuously across those K updates.

`peer_sus` is per-agent (length-N tensor). The platform coupling mirrors
`run_cf_fj.py:372`: the deployed predictor's per-agent prediction blends
with each agent's innate opinion to form the per-round anchor `x_zero`.
With `platform_sus = 0` the platform is absent and `x_zero = innate`
(matches `run_free_fj.py`).

Non-ABM in the sense that there are no per-agent decision rules. Every
agent applies the same fixed-coefficient linear update; the only
heterogeneity is per-agent peer_sus and innate opinion.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld


def normalize_adjacency(adj: Tensor) -> Tensor:
    """Normalize an adjacency matrix the way run_free_fj.py does:

        degs_inv = 1 / col_sum        # (N,)
        W_norm   = adj * degs_inv[:, None]   # broadcast along columns

    The `[:, None]` in numpy reshapes to (N, 1), broadcasting against the
    (N, N) adj to give `W_norm[i, j] = adj[i, j] * degs_inv[i]`. For a
    symmetric (undirected) graph, where row_sum_i == col_sum_i, this
    produces a row-stochastic W: each row of `W_norm` sums to 1, so
    `(W @ x)_i` is a convex combination of `x` and stays in
    `[min(x), max(x)]`. That bounded-mixing property is what makes FJ
    dynamics well-behaved on social graphs.

    Infinite / oversized inverse degrees (isolated nodes) are zeroed.
    """
    degs = adj.sum(dim=0)
    degs_inv = torch.where(degs > 0, 1.0 / degs, torch.zeros_like(degs))
    degs_inv = torch.where(degs_inv > 1.1, torch.zeros_like(degs_inv), degs_inv)
    # `degs_inv.unsqueeze(-1)` is shape (N, 1); multiplies row i by degs_inv[i].
    # `unsqueeze(0)` would be shape (1, N) and would column-normalize instead,
    # which is *not* what run_free_fj does and breaks FJ's bounded-mixing.
    return adj * degs_inv.unsqueeze(-1)


class FJWorld(StatefulPopulationWorld):
    """Linear FJ on a graph with optional platform coupling.

    Use the Simulator's `epoch_size` (or call `world.run(model, K)`
    manually) to evolve the opinion vector through K FJ iterations under
    one model query, matching Algorithm 1 of arxiv 2603.12137.

    Args:
        innate:       (N,) or (N, D) per-agent innate opinion.
        graph:        (N, N) row- or column-normalized influence matrix.
                      Use ``normalize_adjacency`` if you have a raw adjacency.
        peer_sus:     (N,) per-agent peer susceptibility in [0, 1]. A value of
                      ``1`` means the agent does not mix in neighbor opinions
                      (fully stubborn at x_zero); ``0`` means the agent's
                      opinion is the neighbor-weighted average of its peers.
        platform_sus: per-agent platform trust (paper's beta_i). Accepts a
                      length-N Tensor in [0, 1], or a Python scalar that is
                      broadcast to a uniform length-N tensor. 0 across the
                      board means platform-free FJ. Stored as a Tensor.
        features:     optional (N, F) feature matrix passed to the predictor.
                      If None, uses ``innate`` as the feature (matches the
                      "predictor reads innate opinion" baseline).
        initial_opinion: optional (matching innate shape) starting opinion
                      vector. If None, starts from `innate`.
        profiles:     optional row-aligned per-agent metadata, typically a
                      pandas DataFrame (length N). Not used by FJ math; the
                      env carries it so LM-based predictors can look up rich
                      per-agent text features (e.g., age, gender,
                      relation_to_alcohol) via ``world.profiles.iloc[idx]``.
                      The env additionally emits ``data["agent_idx"]`` so the
                      learner can identify which population rows are in a
                      mask-filtered training batch.
    """

    def __init__(
        self,
        innate: Tensor,
        graph: Tensor,
        peer_sus: Tensor,
        *,
        platform_sus: Tensor | float = 0.0,
        features: Tensor | None = None,
        initial_opinion: Tensor | None = None,
        profiles: object | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if innate.ndim not in (1, 2):
            raise ValueError(f"innate must be 1-D or 2-D; got shape {tuple(innate.shape)}")
        n = innate.shape[0]
        if graph.shape != (n, n):
            raise ValueError(f"graph must be ({n}, {n}); got {tuple(graph.shape)}")
        if peer_sus.shape != (n,):
            raise ValueError(f"peer_sus must be (N,); got {tuple(peer_sus.shape)}")

        innate_t = innate.to(dtype=dtype).detach().clone()
        x_init = innate_t.clone() if initial_opinion is None else initial_opinion.to(dtype=dtype).detach().clone()
        if x_init.shape != innate_t.shape:
            raise ValueError(
                f"initial_opinion shape {tuple(x_init.shape)} must match innate "
                f"shape {tuple(innate_t.shape)}"
            )

        # Per-agent platform trust. Accept scalar (broadcast) or (N,) tensor.
        if isinstance(platform_sus, Tensor):
            if platform_sus.shape != (n,):
                raise ValueError(
                    f"platform_sus tensor must be (N,)={(n,)}; got {tuple(platform_sus.shape)}"
                )
            plat = platform_sus.to(dtype=dtype).detach().clone()
        else:
            scalar = float(platform_sus)
            plat = torch.full((n,), scalar, dtype=dtype)
        if not torch.all((plat >= 0.0) & (plat <= 1.0)):
            raise ValueError(
                f"platform_sus must be in [0, 1] per agent; got range "
                f"[{float(plat.min())}, {float(plat.max())}]"
            )

        super().__init__({"opinion": x_init}, dtype=dtype)
        self._innate = innate_t
        self._W = graph.to(dtype=dtype).detach().clone()
        # peer_sus and platform_sus both broadcast against (N,) or (N, D).
        # Store as (N, 1) when innate is 2-D so per-agent values broadcast
        # across the feature dimension.
        ps = peer_sus.to(dtype=dtype).detach().clone()
        self._peer_sus = ps.unsqueeze(-1) if innate_t.ndim == 2 else ps
        self._platform_sus = plat.unsqueeze(-1) if innate_t.ndim == 2 else plat
        self._features = features.to(dtype=dtype).detach().clone() if features is not None else innate_t
        if self._features.shape[0] != n:
            raise ValueError(
                f"features must have N rows ({n}); got {tuple(self._features.shape)}"
            )
        self._n = n
        self._is_scalar = innate_t.ndim == 1

        # Optional row-aligned metadata for LM-based predictors. Not validated
        # for pandas-DataFrame-ness here so the env stays usable without a
        # hard pandas dependency; callers that pass a DataFrame are expected
        # to verify len(profiles) == n themselves, or just match by convention.
        if profiles is not None:
            length = getattr(profiles, "__len__", lambda: -1)()
            if length not in (-1, n):
                raise ValueError(
                    f"profiles length {length} does not match N={n}"
                )
        self._profiles = profiles

        # Per-agent index tensor. Emitted alongside x and y so that
        # Simulator.train_mask filtering yields the labeled subset's indices
        # into the full population, which LM-based learners use to look up
        # the corresponding profile rows.
        self._agent_idx = torch.arange(n)

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
    def platform_sus(self) -> Tensor:
        """Per-agent platform trust. Shape (N,) for scalar innate, (N, 1) for vector innate.

        Returned tensor is a defensive clone; mutating it does not affect the world.
        """
        return self._platform_sus.clone()

    @property
    def profiles(self) -> object | None:
        """Row-aligned per-agent metadata (e.g., a pandas DataFrame), or None."""
        return self._profiles

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

    def _step(self, model: Model) -> tuple[Data, State]:
        """One FJ update that re-queries the model.

        Used by `step` / `sample`. Callers driving an epoch should prefer
        `run(model, n_steps)`, which queries the model exactly once at the
        start of the epoch and iterates the FJ update n_steps times
        without re-querying (matches Algorithm 1 of arxiv 2603.12137).
        """
        predictions = self._predict(model)
        return self._step_with_predictions(predictions)

    def _step_with_predictions(self, predictions: Tensor) -> tuple[Data, State]:
        """One FJ update given pre-computed predictions.

        Internal helper. The platform-trust blending and the FJ averaging
        live here so `run` can call it n_steps times after a single
        model query.
        """
        x_zero = (1.0 - self._platform_sus) * self._innate + self._platform_sus * predictions
        graph_term = self._W @ self._state["opinion"]
        x_new = self._peer_sus * x_zero + (1.0 - self._peer_sus) * graph_term
        # Emit data with a (N, F) feature matrix and (N, 1) label tensor when
        # the underlying innate/opinion is scalar. nn.Linear-style learners
        # require 2-D inputs.
        x_data = self._features.unsqueeze(-1) if self._features.ndim == 1 else self._features
        y = x_new.unsqueeze(-1) if self._is_scalar else x_new
        return {"x": x_data, "y": y, "agent_idx": self._agent_idx.clone()}, {"opinion": x_new}

    def run(self, model: Model, n_steps: int) -> Data:
        """Query the model once for K agent predictions, then iterate the
        FJ update n_steps times under a frozen x_zero.

        Matches Algorithm 1 of arxiv 2603.12137 (Wu, Abebe, Mendler-Dünner):
        the platform is queried once per epoch to set initial conditions;
        the inner peer-mixing dynamics evolve autonomously.
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        predictions = self._predict(model)
        x_zero = (1.0 - self._platform_sus) * self._innate + self._platform_sus * predictions
        one_minus = 1.0 - self._peer_sus
        x = self._state["opinion"]
        for _ in range(n_steps):
            x = self._peer_sus * x_zero + one_minus * (self._W @ x)
        self._state = {"opinion": x}
        x_data = self._features.unsqueeze(-1) if self._features.ndim == 1 else self._features
        y = x.unsqueeze(-1) if self._is_scalar else x
        return {"x": x_data, "y": y, "agent_idx": self._agent_idx.clone()}

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
