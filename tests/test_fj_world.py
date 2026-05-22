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

    def test_platform_sus_per_agent_tensor_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="platform_sus tensor must be"):
            FJWorld(
                innate=torch.zeros(3),
                graph=torch.zeros(3, 3),
                peer_sus=torch.zeros(3),
                platform_sus=torch.tensor([0.1, 0.2]),  # wrong length
            )

    def test_platform_sus_per_agent_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="platform_sus must be"):
            FJWorld(
                innate=torch.zeros(3),
                graph=torch.zeros(3, 3),
                peer_sus=torch.zeros(3),
                platform_sus=torch.tensor([0.1, 1.5, 0.3]),  # one entry too large
            )


class TestPerAgentPlatformTrust:
    """Per-agent platform_sus (paper's beta_i) replaces the legacy scalar."""

    def test_uniform_tensor_matches_scalar(self) -> None:
        """Length-N tensor with all entries = 0.5 should produce the same
        trajectory as scalar platform_sus=0.5."""
        n = 5
        torch.manual_seed(11)
        innate = torch.rand(n)
        W = _random_graph(n, seed=11)
        peer_sus = 0.4 * torch.ones(n)

        class _OneModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(x)

        scalar = FJWorld(innate=innate, graph=W, peer_sus=peer_sus, platform_sus=0.5)
        tensor = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=torch.full((n,), 0.5),
        )
        a = scalar.run(_OneModel(), n_steps=50)
        b = tensor.run(_OneModel(), n_steps=50)
        assert torch.allclose(a["y"], b["y"], atol=1e-6)

    def test_heterogeneous_platform_trust(self) -> None:
        """Two agents with beta_i = 0 should stay near innate; two with
        beta_i = 1 should be pulled toward platform predictions."""
        n = 4
        innate = torch.tensor([0.0, 0.0, 0.0, 0.0])
        W = _random_graph(n, seed=12)
        peer_sus = torch.ones(n)  # fully stubborn at x_zero so peers don't mix

        # Per-agent beta: half ignore platform, half fully trust it.
        platform_sus = torch.tensor([0.0, 0.0, 1.0, 1.0])

        class _OneModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(x)

        world = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=platform_sus,
        )
        data = world.run(_OneModel(), n_steps=200)
        y = data["y"].squeeze(-1)
        # Agents 0, 1: beta=0, peer_sus=1 -> x_zero = innate = 0, stays at 0.
        assert torch.allclose(y[:2], torch.zeros(2), atol=1e-5)
        # Agents 2, 3: beta=1, peer_sus=1 -> x_zero = prediction = 1, stays at 1.
        assert torch.allclose(y[2:], torch.ones(2), atol=1e-5)


class TestSimulatorInitialData:
    """initial_data lets the predictor train at round 0 (Algorithm 1)."""

    def test_initial_data_trains_at_round_zero(self) -> None:
        from perfsim.learners import GradientLearner
        from perfsim.losses import MSELoss
        from perfsim.models import LinearModel
        from perfsim.simulator import Simulator

        n = 8
        torch.manual_seed(13)
        innate = torch.rand(n)
        W = _random_graph(n, seed=13)
        peer_sus = 0.4 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus, platform_sus=0.3)
        model = LinearModel(in_features=1, out_features=1, bias=True)
        loss = MSELoss()
        learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)
        sim = Simulator(world=world, learner=learner, loss=loss)
        init_theta = model.get_params().clone()
        # Round 0 should train on initial_data, so theta should differ from init.
        sim.run(
            n_rounds=1,
            epoch_size=5,
            seed=0,
            initial_data={"x": innate.unsqueeze(-1), "y": innate.unsqueeze(-1)},
        )
        assert not torch.equal(model.get_params(), init_theta)

    def test_initial_data_none_skips_round_zero_training(self) -> None:
        """Without initial_data, the first round leaves theta at its init."""
        from perfsim.learners import GradientLearner
        from perfsim.losses import MSELoss
        from perfsim.models import LinearModel
        from perfsim.simulator import Simulator

        n = 6
        torch.manual_seed(14)
        innate = torch.rand(n)
        W = _random_graph(n, seed=14)
        peer_sus = 0.4 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus, platform_sus=0.3)
        model = LinearModel(in_features=1, out_features=1, bias=True)
        loss = MSELoss()
        learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)
        sim = Simulator(world=world, learner=learner, loss=loss)
        init_theta = model.get_params().clone()
        sim.run(n_rounds=1, epoch_size=5, seed=0)  # no initial_data
        # No training happened (round 0 skipped, round 1 doesn't exist), so
        # theta is unchanged from initialization.
        assert torch.equal(model.get_params(), init_theta)


class TestSimulatorTrainMask:
    """Optional train_mask restricts predictor training to labeled rows."""

    def _setup(self, n: int = 20):
        from perfsim.learners import GradientLearner
        from perfsim.losses import MSELoss
        from perfsim.models import LinearModel
        from perfsim.simulator import Simulator

        torch.manual_seed(20)
        innate = torch.rand(n)
        W = _random_graph(n, seed=20)
        peer_sus = 0.4 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus, platform_sus=0.3)
        model = LinearModel(in_features=1, out_features=1, bias=True)
        loss = MSELoss()
        learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)
        sim = Simulator(world=world, learner=learner, loss=loss)
        return sim, world, innate

    def test_train_mask_filters_rows_seen_by_learner(self) -> None:
        """The learner sees exactly mask.sum() rows when training."""
        from perfsim.core.learner import Learner

        sim, world, innate = self._setup(n=20)

        # Wrap the learner so we can spy on what train() sees.
        rows_seen: list[int] = []
        real_train = sim.predictor.learner.train

        def spy_train(data):
            rows_seen.append(int(data["x"].shape[0]))
            real_train(data)

        sim.predictor.learner.train = spy_train  # type: ignore[method-assign]

        mask = torch.zeros(20, dtype=torch.bool)
        mask[:7] = True  # only first 7 agents labeled

        initial_data = {
            "x": innate.unsqueeze(-1),
            "y": innate.unsqueeze(-1),
        }
        sim.run(
            n_rounds=3,
            epoch_size=2,
            seed=0,
            initial_data=initial_data,
            train_mask=mask,
        )
        # 3 rounds, each trains on 7 rows (mask.sum() == 7).
        assert rows_seen == [7, 7, 7]

    def test_train_mask_unmasked_rows_still_evolve_in_env(self) -> None:
        """Unmasked ("test") agents still participate in FJ dynamics."""
        sim, world, innate = self._setup(n=12)

        mask = torch.zeros(12, dtype=torch.bool)
        mask[:5] = True  # 5 labeled, 7 unlabeled
        initial_data = {"x": innate.unsqueeze(-1), "y": innate.unsqueeze(-1)}
        sim.run(
            n_rounds=2,
            epoch_size=5,
            seed=0,
            initial_data=initial_data,
            train_mask=mask,
        )
        # All 12 agents should have evolved opinions in state["opinion"].
        opinion = world.state["opinion"]
        assert opinion.shape == (12,)
        # Unlabeled agents (indices 5..11) should have moved from their
        # innate values via peer averaging.
        assert not torch.allclose(opinion[5:], innate[5:], atol=1e-6)

    def test_train_mask_validation(self) -> None:
        sim, _, _ = self._setup(n=10)
        with pytest.raises(TypeError, match="bool tensor"):
            sim.run(n_rounds=1, train_mask=torch.zeros(10))  # float dtype
        with pytest.raises(ValueError, match="1-D"):
            sim.run(n_rounds=1, train_mask=torch.zeros(10, 10, dtype=torch.bool))
        with pytest.raises(ValueError, match="zero rows"):
            sim.run(n_rounds=1, train_mask=torch.zeros(10, dtype=torch.bool))


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
            platform_sus=0.0,
        )
        x_star = world.fj_equilibrium(x_zero=innate)
        # One epoch of 500 FJ updates under a single (zero-model) query.
        # Matches Algorithm 1: query model once, evolve internally.
        data = world.run(_ZeroModel(), n_steps=500)
        assert torch.allclose(data["y"].squeeze(-1), x_star, atol=1e-4)

    def test_state_persists(self) -> None:
        n = 6
        torch.manual_seed(1)
        innate = torch.rand(n)
        W = _random_graph(n, seed=1)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(
            innate=innate, graph=W, peer_sus=peer_sus,
            platform_sus=0.0,
        )
        before = world.state["opinion"].clone()
        # One FJ step changes the opinion (since W @ innate != innate for a
        # non-identity graph and non-stubborn peer_sus).
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
            platform_sus=0.5,
        )
        # 200 FJ updates under one model query.
        data = world.run(_ConstantOneModel(), n_steps=200)
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
            platform_sus=0.0,
        )

        class _NoisyModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 100.0 * torch.randn_like(x)

        # With platform_sus=0, the model's output should be ignored: 50
        # FJ steps under a noisy model match 50 FJ steps under the zero model.
        a = free.run(_NoisyModel(), n_steps=50)
        free.reset()
        b = free.run(_ZeroModel(), n_steps=50)
        assert torch.allclose(a["y"], b["y"], atol=1e-5)


class TestModelContract:
    def test_squeezes_singleton_last_dim_for_scalar_innate(self) -> None:
        # LinearModel(in=1, out=1) returns (N, 1); innate is (N,) so we accept.
        n = 4
        innate = torch.rand(n)
        W = _random_graph(n, seed=5)
        peer_sus = 0.5 * torch.ones(n)
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus,
                        platform_sus=0.3)
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
                        platform_sus=0.3)

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
        # epoch_size=20 reproduces the legacy n_ticks=20 inner-loop semantics:
        # 20 FJ updates under fixed theta per outer round, then retrain.
        world = FJWorld(innate=innate, graph=W, peer_sus=peer_sus,
                        platform_sus=0.3)
        model = LinearModel(in_features=1, out_features=1, bias=True)
        learner = ERMLearner(model, MSELoss(), max_iter=20)
        sim = Simulator(world=world, learner=learner, loss=MSELoss())
        history = sim.run(n_rounds=3, epoch_size=20, seed=0)
        assert len(history.records) == 3
        assert not torch.allclose(world.state["opinion"], innate, atol=1e-6)
