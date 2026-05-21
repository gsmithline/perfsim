"""Tests for FJWorld (linear FJ on a graph, torch port of run_free_fj.py)."""

from __future__ import annotations

import pytest
import torch

from perfsim.models import LinearModel
from perfsim.environments.dynamics import FJWorld, normalize_adjacency


def _random_graph(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(n, n, generator=g) < 0.4).float()
    adj.fill_diagonal_(0.0)
    return normalize_adjacency(adj)


class _ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class TestNormalizeAdjacency:
    def test_isolated_node_zeroed(self) -> None:
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0  # nodes 0,1 connected; node 2 isolated
        W = normalize_adjacency(adj)
        # Column 2 was all zeros -> column 2 stays all zeros after column-normalize.
        assert torch.allclose(W[:, 2], torch.zeros(3))


class TestConstructionValidation:
    def test_innate_must_be_1d_or_2d(self) -> None:
        with pytest.raises(ValueError, match="innate must be"):
            FJWorld(
                innate=torch.zeros(2, 3, 4),
                graph=torch.zeros(2, 2),
                peer_sus=torch.zeros(2),
            )

    def test_graph_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="graph must be"):
            FJWorld(
                innate=torch.zeros(5),
                graph=torch.zeros(4, 4),
                peer_sus=torch.zeros(5),
            )

    def test_peer_sus_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="peer_sus must be"):
            FJWorld(
                innate=torch.zeros(5),
                graph=torch.zeros(5, 5),
                peer_sus=torch.zeros(4),
            )

    def test_platform_sus_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="platform_sus must be"):
            FJWorld(
                innate=torch.zeros(3),
                graph=torch.zeros(3, 3),
                peer_sus=torch.zeros(3),
                platform_sus=1.5,
            )


class TestFreeFJBaseline:
    """With platform_sus=0, the world matches run_free_fj.py exactly:
    x_zero = innate every round; inner loop reaches the FJ equilibrium."""

    def test_long_rollout_matches_closed_form(self) -> None:
        n = 8
        torch.manual_seed(0)
        innate = torch.rand(n) * 2 - 1
        W = _random_graph(n, seed=0)
        # Heterogeneous peer_sus (matches the source script's hetero_peer_sus pkl).
        peer_sus = 0.3 + 0.4 * torch.rand(n)
        world = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=0.0, n_ticks=500,
        )
        x_star = world.fj_equilibrium(x_zero=innate)
        data = world.sample(_ZeroModel())
        assert torch.allclose(data["y"].squeeze(-1), x_star, atol=1e-4)

    def test_state_persists(self) -> None:
        n = 6
        torch.manual_seed(1)
        innate = torch.rand(n)
        W = _random_graph(n, seed=1)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=0.0, n_ticks=20,
        )
        before = world.state["opinion"].clone()
        world.step(_ZeroModel())
        assert not torch.allclose(world.state["opinion"], before, atol=1e-6)

    def test_reset_restores_initial(self) -> None:
        n = 4
        innate = torch.tensor([0.1, 0.4, -0.2, 0.3])
        W = _random_graph(n, seed=2)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus)
        world.step(_ZeroModel())
        world.step(_ZeroModel())
        before = world.state["opinion"].clone()
        world.reset()
        # After reset, state should match initial_opinion = innate.
        assert torch.allclose(world.state["opinion"], innate, atol=1e-6)
        assert not torch.allclose(world.state["opinion"], before, atol=1e-6)


class TestPlatformCoupling:
    """With platform_sus > 0, the predictor influences x_zero."""

    def test_platform_pulls_anchor(self) -> None:
        n = 5
        innate = torch.zeros(n)  # innate at 0
        W = _random_graph(n, seed=3)
        peer_sus = 0.4 * torch.ones(n)

        class _ConstantOneModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(x)

        # platform_sus=0.5, predictions=1 -> x_zero = 0.5 across the board.
        world = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=0.5, n_ticks=200,
        )
        data = world.sample(_ConstantOneModel())
        # FJ equilibrium with constant x_zero=0.5 and row-stochastic-ish W is 0.5
        # in the limit (anchor dominates as the system equilibrates).
        # With column-normalized W from normalize_adjacency the limit isn't
        # exactly 0.5, but it should be far from 0 (the innate) and close to 0.5.
        mean = data["y"].mean().item()
        assert abs(mean - 0.5) < 0.1
        assert mean > 0.1  # definitely moved off innate=0

    def test_zero_platform_sus_equals_free_baseline(self) -> None:
        n = 6
        innate = torch.linspace(-1, 1, n)
        W = _random_graph(n, seed=4)
        peer_sus = 0.5 * torch.ones(n)
        # Both worlds have the same state; only platform_sus differs.
        free = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=0.0, n_ticks=50,
        )

        class _NoisyModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 100.0 * torch.randn_like(x)

        # With platform_sus=0, the model's output should be ignored.
        a = free.sample(_NoisyModel())
        free.reset()
        b = free.sample(_ZeroModel())
        assert torch.allclose(a["y"], b["y"], atol=1e-5)


class TestModelContract:
    def test_squeezes_singleton_last_dim_for_scalar_innate(self) -> None:
        # LinearModel(in=1, out=1) returns (N, 1); innate is (N,) so we accept.
        n = 4
        innate = torch.rand(n)
        W = _random_graph(n, seed=5)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus,
                        platform_sus=0.3, n_ticks=20)
        model = LinearModel(in_features=1, out_features=1, bias=False)
        model.set_params(torch.tensor([0.5]))
        data = world.sample(model)
        assert data["y"].shape == (n, 1)

    def test_rejects_wrong_output_shape(self) -> None:
        n = 4
        innate = torch.rand(n)
        W = _random_graph(n, seed=6)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus,
                        platform_sus=0.3, n_ticks=10)

        class _WrongShape(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(n, 3)

        with pytest.raises(ValueError, match="model output shape"):
            world.sample(_WrongShape())


class TestSimulatorIntegration:
    def test_runs_through_simulator(self) -> None:
        from perfsim.learners import ERMLearner
        from perfsim.losses import MSELoss
        from perfsim.simulator import Simulator

        n = 10
        torch.manual_seed(7)
        innate = torch.rand(n) * 2 - 1
        W = _random_graph(n, seed=7)
        peer_sus = 0.4 * torch.ones(n)
        # Use scalar innate; predictor maps innate -> opinion prediction.
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus,
                        platform_sus=0.3, n_ticks=20)
        model = LinearModel(in_features=1, out_features=1, bias=True)
        learner = ERMLearner(model, MSELoss(), max_iter=20)
        sim = Simulator(world=world, learner=learner, loss=MSELoss())
        history = sim.run(n_rounds=3, seed=0)
        assert len(history.records) == 3
        assert not torch.allclose(world.state["opinion"], innate, atol=1e-6)
