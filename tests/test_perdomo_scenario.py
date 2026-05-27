"""End-to-end tests for the Perdomo loan scenario (synthetic fallback path).

The real-data path requires Kaggle credentials and is documented in
`scenarios/perdomo_loan/README.md`; we do not exercise it in CI. Tests
here cover:
- synthetic dataset construction is deterministic and has expected shape.
- build_world standardizes and applies Perdomo's negative-epsilon convention.
- run() executes end-to-end on synthetic data and produces History.
- ERM and Gradient learners both run; PR is recorded.
- Config content_hash is deterministic.
"""

from __future__ import annotations

import pytest
import torch

from perfsim.datasets import InMemoryDataset
from perfsim.learners import ERMLearner
from perfsim.losses import BCEWithLogitsLoss
from perfsim.models import LinearModel
from perfsim.scenarios.perdomo_loan import (
    PerdomoLoanConfig,
    build_dataset,
    build_world,
    make_synthetic_dataset,
    run,
)
from perfsim.scenarios.perdomo_loan.config import PERDOMO_STRAT_FEATURES, _balance_classes
from perfsim.simulator import Simulator


class TestSyntheticDataset:
    def test_shape(self) -> None:
        ds = make_synthetic_dataset(n=200, d=8, seed=0)
        data = ds.load()
        assert data["x"].shape == (200, 8)
        assert data["y"].shape == (200,)

    def test_deterministic(self) -> None:
        a = make_synthetic_dataset(n=100, d=5, seed=42)
        b = make_synthetic_dataset(n=100, d=5, seed=42)
        assert a.hash() == b.hash()

    def test_seed_changes_data(self) -> None:
        a = make_synthetic_dataset(n=100, d=5, seed=0)
        b = make_synthetic_dataset(n=100, d=5, seed=1)
        assert a.hash() != b.hash()

    def test_labels_are_binary(self) -> None:
        ds = make_synthetic_dataset(n=500, seed=0)
        y = ds.load()["y"]
        assert set(y.unique().tolist()) <= {0.0, 1.0}


class TestBuildWorld:
    def test_dim_matches_dataset(self) -> None:
        ds = make_synthetic_dataset(n=100, d=7, seed=0)
        world = build_world(ds, mu=10.0, standardize=False)
        assert world.dim == 7
        assert world.n_agents == 100

    def test_negative_mu_rejected(self) -> None:
        ds = make_synthetic_dataset(n=10, d=3, seed=0)
        with pytest.raises(ValueError, match="mu must be >= 0"):
            build_world(ds, mu=-1.0)

    def test_standardize_zero_mean_unit_var(self) -> None:
        ds = make_synthetic_dataset(n=500, d=4, seed=0)
        world = build_world(ds, mu=0.0, standardize=True, robust=False, clip=0.0)
        x0 = world._x0
        assert torch.allclose(x0.mean(dim=0), torch.zeros(4), atol=1e-5)
        assert torch.allclose(x0.std(dim=0), torch.ones(4), atol=1e-3)

    def test_robust_standardize_centers_at_median(self) -> None:
        ds = make_synthetic_dataset(n=500, d=4, seed=0)
        world = build_world(ds, mu=0.0, standardize=True, robust=True, clip=0.0)
        x0 = world._x0
        assert torch.allclose(x0.median(dim=0).values, torch.zeros(4), atol=1e-5)

    def test_clip_bounds_features(self) -> None:
        x = torch.randn(100, 3)
        x[0] = 1000.0
        y = torch.zeros(100)
        ds = InMemoryDataset({"x": x, "y": y})
        world = build_world(ds, mu=0.0, standardize=True, robust=True, clip=5.0)
        assert world._x0.abs().max().item() <= 5.0 + 1e-6

    def test_clip_zero_disables(self) -> None:
        x = torch.randn(100, 3)
        x[0] = 1000.0
        y = torch.zeros(100)
        ds = InMemoryDataset({"x": x, "y": y})
        world = build_world(ds, mu=0.0, standardize=True, robust=True, clip=0.0)
        assert world._x0.abs().max().item() > 100.0

    def test_negative_clip_rejected(self) -> None:
        ds = make_synthetic_dataset(n=10, d=3, seed=0)
        with pytest.raises(ValueError, match="clip must be >= 0"):
            build_world(ds, mu=0.0, clip=-1.0)

    def test_perdomo_minus_convention(self) -> None:
        """At mu=1, shift should be -theta (not +theta)."""
        ds = make_synthetic_dataset(n=10, d=4, seed=0)
        world = build_world(ds, mu=1.0, standardize=False)
        model = LinearModel(in_features=4, out_features=1, bias=True)
        model.linear.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        data = world.sample(model)
        x0 = world._x0
        expected = x0 - torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(data["x"], expected)

    def test_mu_zero_no_shift(self) -> None:
        ds = make_synthetic_dataset(n=10, d=4, seed=0)
        world = build_world(ds, mu=0.0, standardize=False)
        model = LinearModel(in_features=4, out_features=1, bias=True)
        model.linear.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        data = world.sample(model)
        assert torch.allclose(data["x"], world._x0)

    def test_strat_features_only_shifts_those_columns(self) -> None:
        ds = make_synthetic_dataset(n=10, d=5, seed=0)
        world = build_world(
            ds, mu=1.0, standardize=False, strat_features=(0, 2, 4)
        )
        model = LinearModel(in_features=5, out_features=1, bias=True)
        model.linear.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        data = world.sample(model)
        # Only columns 0, 2, 4 should be shifted by -theta; cols 1, 3 stay
        x0 = world._x0
        diff = data["x"] - x0
        assert torch.allclose(diff[:, 0], torch.full((10,), -1.0))
        assert torch.allclose(diff[:, 1], torch.zeros(10))
        assert torch.allclose(diff[:, 2], torch.full((10,), -3.0))
        assert torch.allclose(diff[:, 3], torch.zeros(10))
        assert torch.allclose(diff[:, 4], torch.full((10,), -5.0))


class TestRunSynthetic:
    def test_runs_end_to_end_erm(self) -> None:
        config = PerdomoLoanConfig(
            mu=1.0,
            n_rounds=5,
            learner="erm",
            use_synthetic_fallback=True,
            synthetic_n=200,
            synthetic_d=10,
            seed=0,
        )
        history = run(config)
        assert len(history) == 5
        for r in history.records:
            assert "theta" in r
            assert "PR" in r
            assert torch.is_tensor(r["PR"])

    def test_runs_end_to_end_gradient(self) -> None:
        config = PerdomoLoanConfig(
            mu=1.0,
            n_rounds=4,
            learner="gradient",
            learner_lr=0.05,
            use_synthetic_fallback=True,
            synthetic_n=200,
            synthetic_d=10,
            seed=0,
        )
        history = run(config)
        assert len(history) == 4

    def test_unknown_learner_rejected(self) -> None:
        config = PerdomoLoanConfig(
            mu=1.0,
            n_rounds=2,
            learner="bogus",
            use_synthetic_fallback=True,
            synthetic_n=50,
            synthetic_d=4,
            strat_features=(0, 1, 2),
        )
        with pytest.raises(ValueError, match="unknown learner"):
            run(config)

    def test_dataset_hash_recorded(self) -> None:
        config = PerdomoLoanConfig(
            mu=1.0,
            n_rounds=3,
            use_synthetic_fallback=True,
            synthetic_n=100,
            synthetic_d=10,
            seed=0,
        )
        history = run(config)
        for r in history.records:
            assert "dataset_hash" in r

    def test_determinism_across_runs(self) -> None:
        config = PerdomoLoanConfig(
            mu=1.0,
            n_rounds=4,
            use_synthetic_fallback=True,
            synthetic_n=100,
            synthetic_d=10,
            seed=42,
        )
        h1 = run(config)
        h2 = run(config)
        for r1, r2 in zip(h1.records, h2.records):
            assert torch.allclose(r1["theta"], r2["theta"], atol=1e-6)


class TestConfig:
    def test_default_targets_real_data(self) -> None:
        cfg = PerdomoLoanConfig()
        assert cfg.use_synthetic_fallback is False
        assert cfg.balance_classes is True
        assert cfg.decay_bias is False

    def test_default_strat_features_is_perdomo(self) -> None:
        cfg = PerdomoLoanConfig()
        assert cfg.strat_features == PERDOMO_STRAT_FEATURES
        assert cfg.strat_features == (0, 5, 7)

    def test_content_hash_deterministic(self) -> None:
        a = PerdomoLoanConfig(mu=10.0, seed=0)
        b = PerdomoLoanConfig(mu=10.0, seed=0)
        assert a.content_hash() == b.content_hash()

    def test_content_hash_sensitive_to_mu(self) -> None:
        a = PerdomoLoanConfig(mu=1.0)
        b = PerdomoLoanConfig(mu=2.0)
        assert a.content_hash() != b.content_hash()

    def test_content_hash_sensitive_to_strat_features(self) -> None:
        a = PerdomoLoanConfig(strat_features=(0, 5, 7))
        b = PerdomoLoanConfig(strat_features=(0, 1, 2))
        assert a.content_hash() != b.content_hash()

    def test_build_dataset_synthetic_path(self) -> None:
        cfg = PerdomoLoanConfig(use_synthetic_fallback=True, synthetic_n=20, synthetic_d=3, seed=0)
        ds = build_dataset(cfg)
        assert isinstance(ds, InMemoryDataset)
        assert ds.load()["x"].shape == (20, 3)


class TestClassBalancing:
    def _make_imbalanced(self, n_pos: int, n_neg: int) -> "InMemoryDataset":
        x = torch.cat([torch.ones(n_pos, 3), torch.zeros(n_neg, 3)])
        y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
        return InMemoryDataset({"x": x, "y": y})

    def test_balance_takes_all_positives_plus_capped_negatives(self) -> None:
        raw = self._make_imbalanced(n_pos=50, n_neg=1000)
        bal = _balance_classes(raw, n_negatives=10, seed=0)
        data = bal.load()
        y = data["y"]
        assert (y == 1.0).sum().item() == 50
        assert (y == 0.0).sum().item() == 10

    def test_balance_deterministic_with_seed(self) -> None:
        raw = self._make_imbalanced(n_pos=30, n_neg=500)
        a = _balance_classes(raw, n_negatives=10, seed=42)
        b = _balance_classes(raw, n_negatives=10, seed=42)
        assert a.hash() == b.hash()

    def test_balance_seed_changes_order(self) -> None:
        raw = self._make_imbalanced(n_pos=30, n_neg=500)
        a = _balance_classes(raw, n_negatives=10, seed=0)
        b = _balance_classes(raw, n_negatives=10, seed=1)
        # Same content but different order; loaded tensors differ -> different hash
        assert a.hash() != b.hash()

    def test_n_negatives_fewer_than_available(self) -> None:
        raw = self._make_imbalanced(n_pos=10, n_neg=5)
        bal = _balance_classes(raw, n_negatives=100, seed=0)  # asks for more than exists
        y = bal.load()["y"]
        assert (y == 0.0).sum().item() == 5


class TestInMemoryDataset:
    def test_round_trip_hash(self) -> None:
        data = {"x": torch.randn(10, 4), "y": torch.zeros(10)}
        a = InMemoryDataset(data)
        b = InMemoryDataset(data)
        assert a.hash() == b.hash()

    def test_clone_independence(self) -> None:
        x = torch.randn(10, 4)
        y = torch.zeros(10)
        ds = InMemoryDataset({"x": x, "y": y})
        # Mutating the original tensor must not affect the dataset
        x.zero_()
        loaded_x = ds.load()["x"]
        assert not torch.allclose(loaded_x, torch.zeros_like(loaded_x))
