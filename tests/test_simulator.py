"""Unit tests for Simulator: binding, run loop, history, metrics callbacks,
determinism, dataset-hash recording.

Includes a determinism check that doubles as validation test 4 ((config,
seed, dataset hash) determinism).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perfsim.core import SchemaError
from perfsim.core.dataset import Dataset
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.core.world import StatelessWorld
from perfsim.datasets import TensorDataset
from perfsim.learners import ERMLearner, GradientLearner
from perfsim.losses import MSELoss
from perfsim.metrics import performative_risk
from perfsim.models import LinearModel
from perfsim.simulator import Simulator
from perfsim.worlds import GaussianShiftWorld


def _make_sim(
    *,
    d: int = 3,
    sigma: float = 0.005,
    batch: int = 256,
    learner_cls: type = ERMLearner,
    metrics: dict[str, Any] | None = None,
    dataset: Dataset | None = None,
) -> tuple[Simulator, LinearModel, GaussianShiftWorld]:
    A = 0.5 * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d].clone()
    world = GaussianShiftWorld(A=A, b=b, sigma_noise=sigma, batch_size=batch)
    model = LinearModel(in_features=d, out_features=1, bias=False)
    loss = MSELoss()
    if learner_cls is ERMLearner:
        learner = ERMLearner(model, loss, max_iter=100)
    else:
        learner = GradientLearner(model, loss, lr=0.05, steps_per_round=1)
    sim = Simulator(
        world=world, learner=learner, loss=loss, metrics=metrics, dataset=dataset
    )
    return sim, model, world


class TestBinding:
    def test_compatible_pair_binds(self) -> None:
        # Should not raise
        sim, _, _ = _make_sim()
        assert sim is not None

    def test_incompatible_pair_rejected(self) -> None:
        # Build a Learner that only accepts a nonexistent schema
        class _PickyLearner(ERMLearner):
            accepted_schemas = (
                DataSchema(name="picky", required=frozenset({"q", "z"})),
            )

        A = 0.5 * torch.eye(3)
        b = torch.tensor([1.0, 0.5, -0.5])
        world = GaussianShiftWorld(A=A, b=b)
        model = LinearModel(in_features=3, out_features=1, bias=False)
        loss = MSELoss()
        learner = _PickyLearner(model, loss)
        with pytest.raises(SchemaError, match="does not accept"):
            Simulator(world=world, learner=learner, loss=loss)


class TestRunLoop:
    def test_run_populates_history(self) -> None:
        sim, _, _ = _make_sim()
        history = sim.run(n_rounds=5, seed=0)
        assert len(history) == 5
        for t, record in enumerate(history.records):
            assert record["round"] == t
            assert "theta" in record

    def test_stability_gap_recorded_after_first_round(self) -> None:
        sim, _, _ = _make_sim()
        history = sim.run(n_rounds=3, seed=0)
        assert "stability_gap" not in history[0]
        assert "stability_gap" in history[1]
        assert "stability_gap" in history[2]

    def test_metrics_callbacks_invoked(self) -> None:
        def pr(sim: Simulator) -> torch.Tensor:
            return performative_risk(sim.world, sim.learner.model, sim.loss)

        sim, _, _ = _make_sim(metrics={"pr": pr})
        history = sim.run(n_rounds=4, seed=0)
        for record in history.records:
            assert "pr" in record
            assert torch.is_tensor(record["pr"])

    def test_current_round_updated(self) -> None:
        sim, _, _ = _make_sim()
        assert sim.current_round == -1
        sim.run(n_rounds=3, seed=0)
        assert sim.current_round == 2  # last round


class TestDeterminism:
    """Validation test 4 (gating): (config, seed) -> deterministic trajectory."""

    def test_same_seed_same_trajectory_erm(self) -> None:
        sim_a, _, _ = _make_sim()
        h_a = sim_a.run(n_rounds=8, seed=42)
        sim_b, _, _ = _make_sim()
        h_b = sim_b.run(n_rounds=8, seed=42)
        for ra, rb in zip(h_a.records, h_b.records):
            assert torch.allclose(ra["theta"], rb["theta"], atol=1e-6)

    def test_same_seed_same_trajectory_gradient(self) -> None:
        sim_a, _, _ = _make_sim(learner_cls=GradientLearner)
        h_a = sim_a.run(n_rounds=10, seed=7)
        sim_b, _, _ = _make_sim(learner_cls=GradientLearner)
        h_b = sim_b.run(n_rounds=10, seed=7)
        for ra, rb in zip(h_a.records, h_b.records):
            assert torch.allclose(ra["theta"], rb["theta"], atol=1e-6)

    def test_different_seed_differs(self) -> None:
        sim_a, _, _ = _make_sim()
        h_a = sim_a.run(n_rounds=4, seed=42)
        sim_b, _, _ = _make_sim()
        h_b = sim_b.run(n_rounds=4, seed=43)
        # Different seeds must produce non-identical trajectories. ERM converges
        # fast on this toy, so per-component differences may be small (sample
        # noise scale); we just check the trajectories are not bitwise equal.
        assert not torch.equal(h_a[-1]["theta"], h_b[-1]["theta"])


class TestDatasetHash:
    def test_hash_recorded_when_dataset_provided(self, tmp_path: Path) -> None:
        p = tmp_path / "d.npz"
        rng = np.random.default_rng(0)
        np.savez(p, x=rng.standard_normal((20, 3)).astype(np.float32), y=rng.standard_normal(20).astype(np.float32))
        dataset = TensorDataset(p)
        expected = dataset.hash()
        sim, _, _ = _make_sim(dataset=dataset)
        history = sim.run(n_rounds=2, seed=0)
        for record in history.records:
            assert record["dataset_hash"] == expected

    def test_no_hash_recorded_without_dataset(self) -> None:
        sim, _, _ = _make_sim()
        history = sim.run(n_rounds=2, seed=0)
        for record in history.records:
            assert "dataset_hash" not in record


class TestEndToEnd:
    def test_convergence_via_simulator(self) -> None:
        """ERM via Simulator converges to closed-form FP on GaussianShiftWorld."""
        sim, model, world = _make_sim(d=3, sigma=0.005, batch=1024)
        sim.run(n_rounds=30, seed=0)
        fp = world.closed_form_fp()
        assert torch.allclose(model.get_params(), fp, atol=0.05)
