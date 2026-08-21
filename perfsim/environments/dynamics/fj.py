"""FJWorld: linear Friedkin-Johnsen opinion dynamics on a graph.

One step blends each agent's innate opinion with the deployed model's
prediction (weighted by platform_sus) to form an anchor x_zero, then mixes
that anchor with the neighbor-weighted average (weighted by peer_sus). With
platform_sus=0 the platform is absent and x_zero=innate. Non-ABM: every agent
applies the same linear update, heterogeneity is only per-agent peer_sus/innate.
"""

from __future__ import annotations

import torch
from torch import Tensor

from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.environments.dynamics.stateful_population import State, StatefulPopulationWorld


def normalize_adjacency(adj: Tensor) -> Tensor:
    """Row-normalize an adjacency matrix into a row-stochastic W (FJ mixing).

    Mirrors run_free_fj.py: W[i,j] = adj[i,j] / col_sum[j], via row-wise
    multiply by 1/deg. Isolated / oversized inverse degrees are zeroed.
    """
    degs = adj.sum(dim=0)
    degs_inv = torch.where(degs > 0, 1.0 / degs, torch.zeros_like(degs))
    degs_inv = torch.where(degs_inv > 1.1, torch.zeros_like(degs_inv), degs_inv)
    # unsqueeze(-1) -> (N,1) row-normalizes; unsqueeze(0) -> (1,N) would
    return adj * degs_inv.unsqueeze(-1)


class FJWorld(StatefulPopulationWorld):
    """Linear FJ on a graph with optional platform coupling.

    Args:
        innate:       (N,) or (N, D) per-agent innate opinion.
        graph:        (N, N) normalized influence matrix (see normalize_adjacency).
        peer_sus:     (N,) peer susceptibility in [0, 1]. 1 = stubborn at x_zero,
                      0 = pure neighbor average.
        platform_sus: per-agent platform trust (paper's beta_i), (N,) tensor or
                      scalar broadcast. 0 = platform-free FJ.
        features:     optional (N, F) features for the predictor; defaults to innate.
        initial_opinion: optional starting opinion (innate shape); defaults to innate.
        profiles:     optional row-aligned per-agent metadata for LM predictors;
                      emitted indices come via data["agent_idx"].
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

        if isinstance(platform_sus, Tensor):
            if platform_sus.shape != (n,):
                raise ValueError(
                    f"platform_sus tensor must be (N,)={(n,)}; got {tuple(platform_sus.shape)}"
                )
            plat = platform_sus.to(dtype=dtype).detach().clone()
        else:
            plat = torch.full((n,), float(platform_sus), dtype=dtype)
        if not torch.all((plat >= 0.0) & (plat <= 1.0)):
            raise ValueError(
                f"platform_sus must be in [0, 1] per agent; got range "
                f"[{float(plat.min())}, {float(plat.max())}]"
            )

        super().__init__({"opinion": x_init}, dtype=dtype)
        self._innate = innate_t
        self._W = graph.to(dtype=dtype).detach().clone()
        # Store per-agent coefficients as (N,1) for 2-D innate so they
        # broadcast across the feature dimension.
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

        if profiles is not None:
            length = getattr(profiles, "__len__", lambda: -1)()
            if length not in (-1, n):
                raise ValueError(f"profiles length {length} does not match N={n}")
        self._profiles = profiles
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
        """Per-agent platform trust, (N,) or (N, 1). Defensive clone."""
        return self._platform_sus.clone()

    @property
    def current_opinion(self) -> Tensor:
        return self._state["opinion"].clone()

    @property
    def profiles(self) -> object | None:
        """Row-aligned per-agent metadata (e.g. a pandas DataFrame), or None."""
        return self._profiles

    def _predict(self, model: Model) -> Tensor:
        feat = self._features
        if feat.ndim == 1:
            feat = feat.unsqueeze(-1)
        with torch.no_grad():
            r = model(feat)
        if r.shape[0] != self._n:
            raise ValueError(
                f"model output leading dim {r.shape[0]} does not match N={self._n}"
            )
        if self._is_scalar and r.ndim == 2 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        if r.shape != self._innate.shape:
            raise ValueError(
                f"model output shape {tuple(r.shape)} does not match innate "
                f"shape {tuple(self._innate.shape)}"
            )
        return r.detach().to(self._dtype)

    def _step(self, model: Model) -> tuple[Data, State]:
        """One FJ update that re-queries the model. Prefer `run` for epochs."""
        predictions = self._predict(model)
        return self._step_with_predictions(predictions)

    def _step_with_predictions(self, predictions: Tensor) -> tuple[Data, State]:
        """One FJ update given pre-computed predictions."""
        x_zero = (1.0 - self._platform_sus) * self._innate + self._platform_sus * predictions
        graph_term = self._W @ self._state["opinion"]
        x_new = self._peer_sus * x_zero + (1.0 - self._peer_sus) * graph_term
        x_data = self._features.unsqueeze(-1) if self._features.ndim == 1 else self._features
        y = x_new.unsqueeze(-1) if self._is_scalar else x_new
        return {"x": x_data, "y": y, "agent_idx": self._agent_idx.clone()}, {"opinion": x_new}

    def run(self, model: Model, n_steps: int) -> Data:
        """Query the model once, then iterate the FJ update n_steps times.

        Matches Algorithm 1 of arxiv 2603.12137: platform queried once per
        epoch to set x_zero, inner peer-mixing evolves autonomously.
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

    # ------------------------------------------------------------------
    # WU-STYLE FJ UPDATE (fj_update_version="wu1", 2026-08-21)
    #
    # OPT-IN AND ADDITIVE. `run()` above is the archived operator and is
    # left exactly as it was; every existing FJ artifact replays through
    # it unchanged. This method is reached only when the runner is asked
    # for it explicitly, so no historical result silently changes.
    #
    # It differs from `run()` in three ways, each a documented blocker of
    # the archived path:
    #
    #   1  IT DOES NOT QUERY THE MODEL. The served vector is passed in.
    #      `run()` calls _predict() itself, while the runner separately
    #      records its own served vector in pred_raw -- so the vector
    #      applied to the population was not guaranteed to be the vector
    #      on record, and generation happened twice per round.
    #
    #   2  THE INNER LOOP STARTS AT THE NEW ANCHOR, x^0_inner(t) = x^0(t).
    #      `run()` starts from the PREVIOUS round's state, which carries
    #      population memory across rounds and is not the Wu recurrence.
    #      Starting at the anchor is what makes the human component
    #      stateless (k = 1): the only channel from round t-1 to round t
    #      is the model, never the population.
    #
    #   3  alpha IS NAMED, AND THE COMPLEMENT IS ENFORCED. The stored
    #      field `peer_sus` is STUBBORNNESS, i.e. the anchor weight, so
    #      the peer susceptibility is alpha = 1 - peer_sus. MovieLens
    #      ships peer_sus = 1, which under the archived operator means
    #      FULLY ANCHORED -- no neighbour mixing at all. Passing alpha
    #      straight into the internal coefficient is therefore the single
    #      most damaging mistake available here: at alpha = .9 it would
    #      run a stubbornness of .9, i.e. a peer susceptibility of .1,
    #      the near-opposite of the intended dynamics. This method
    #      REFUSES that: the caller passes the peer susceptibility, and
    #      run_wu checks the world was constructed with its complement.
    #
    # Jiduan Wu's model, homogeneous-parameter specialization:
    #     x_init^(t)  = (1 - beta) x^innate + beta m_t
    #     u^(0)       = x_init^(t)
    #     u^(l+1)     = (1 - alpha) x_init^(t) + alpha P u^(l),
    #                   l = 0 .. K-1
    # with beta carried per agent in platform_sus (the runner folds
    # W_PLAT and PLATFORM_SUS_SCALE into it) and P = self._W.
    # ------------------------------------------------------------------
    def run_wu(self, predictions: Tensor, *, alpha: float,
               n_inner: int = 1) -> Data:
        """One outer round of Wu's FJ update from a SERVED vector.

        `alpha` is the PEER SUSCEPTIBILITY. The world must have been
        constructed with peer_sus = 1 - alpha (stubbornness); that is
        checked here rather than trusted, because passing alpha directly
        into the internal coefficient silently runs the complement of the
        intended dynamics and every downstream number would still look
        well-formed.

        Returns the emitted Data and leaves `state["opinion"]` at u^(K).
        """
        if not isinstance(n_inner, int) or n_inner < 1:
            raise ValueError(f"n_inner must be a positive int; got {n_inner!r}")
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {alpha!r}")
        want_stub = 1.0 - float(alpha)
        stub = self._peer_sus.reshape(-1)
        if not torch.allclose(stub, torch.full_like(stub, want_stub),
                              atol=1e-6):
            raise ValueError(
                f"run_wu: peer susceptibility alpha={alpha} requires the "
                f"world to carry stubbornness peer_sus={want_stub}, but it "
                f"holds [{float(stub.min())}, {float(stub.max())}]. The "
                f"internal coefficient is STUBBORNNESS, not alpha -- "
                f"passing alpha straight in runs 1-alpha peer mixing.")
        preds = predictions.to(dtype=self._dtype).detach()
        if self._is_scalar and preds.ndim == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if preds.shape != self._innate.shape:
            raise ValueError(
                f"predictions shape {tuple(preds.shape)} must match innate "
                f"shape {tuple(self._innate.shape)}")
        if not torch.isfinite(preds).all():
            raise ValueError("predictions contain non-finite values")

        a = float(alpha)
        x_init = ((1.0 - self._platform_sus) * self._innate
                  + self._platform_sus * preds)
        u = x_init                      # u^(0) = x_init^(t), NOT the
        u1 = None                       # previous round's population
        for _ in range(n_inner):
            u = (1.0 - a) * x_init + a * (self._W @ u)
            if u1 is None:
                u1 = u.clone()
        # THE FIRST INNER ITERATE IS KEPT AS EVIDENCE.
        # At alpha=.9 and K=100 the inner loop contracts by 0.9^100 ~ 3e-5,
        # so u^(K) is the FJ fixed point and has FORGOTTEN where it
        # started. Replaying the final state therefore cannot distinguish
        # u^(0)=x_init from a stale previous-population start -- the very
        # blocker this operator exists to fix would be unfalsifiable from
        # the artifact. u^(1) still carries it: u^(1) is an affine
        # function of u^(0) with coefficient alpha, so pinning u^(1) pins
        # the initialisation exactly.
        self.last_u1 = u1
        self.last_x_init = x_init.clone()
        x = u
        self._state = {"opinion": x}
        x_data = (self._features.unsqueeze(-1) if self._features.ndim == 1
                  else self._features)
        y = x.unsqueeze(-1) if self._is_scalar else x
        return {"x": x_data, "y": y, "agent_idx": self._agent_idx.clone()}

    @staticmethod
    def alpha_to_peer_sus(alpha: float) -> float:
        """Peer susceptibility alpha -> the STUBBORNNESS this class stores.

        Stated once, in the class that owns the convention, instead of
        being re-derived wherever alpha is used. run_wu enforces it.
        """
        return 1.0 - float(alpha)

    def fj_equilibrium(self, x_zero: Tensor | None = None) -> Tensor:
        """Closed-form FJ fixed point: solve (I - diag(1-a) W) x* = diag(a) x_zero.

        Default x_zero = innate (the platform-free anchor).
        """
        if x_zero is None:
            x_zero = self._innate
        ps = self._peer_sus.reshape(-1)
        eye = torch.eye(self._n, dtype=self._dtype)
        A = eye - (1.0 - ps).unsqueeze(-1) * self._W
        if self._is_scalar:
            rhs = ps * x_zero
            return torch.linalg.solve(A, rhs)
        rhs = ps.unsqueeze(-1) * x_zero
        return torch.linalg.solve(A, rhs)
