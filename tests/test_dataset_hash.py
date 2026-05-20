"""Validation test 6: Dataset.hash() is deterministic; replays with mismatched
hashes fail loudly.

Also covers TensorDataset load/split mechanics built alongside the Dataset
ABC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from perfsim.core import Dataset, SchemaError
from perfsim.core.types import DataSchema
from perfsim.datasets import TensorDataset


def _make_npz(path: Path, *, n: int = 100, d: int = 4, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    np.savez(path, x=x, y=y)
    return path


class TestHashDeterminism:
    def test_same_file_same_hash(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz")
        assert TensorDataset(p).hash() == TensorDataset(p).hash()

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        p1 = tmp_path / "d1.npz"
        p2 = tmp_path / "d2.npz"
        np.savez(p1, x=np.zeros((10, 3), dtype=np.float32), y=np.zeros(10, dtype=np.int64))
        np.savez(p2, x=np.ones((10, 3), dtype=np.float32), y=np.zeros(10, dtype=np.int64))
        assert TensorDataset(p1).hash() != TensorDataset(p2).hash()

    def test_different_dtype_different_hash(self, tmp_path: Path) -> None:
        p1 = tmp_path / "d1.npz"
        p2 = tmp_path / "d2.npz"
        np.savez(p1, x=np.zeros((10, 3), dtype=np.float32), y=np.zeros(10, dtype=np.int64))
        np.savez(p2, x=np.zeros((10, 3), dtype=np.float64), y=np.zeros(10, dtype=np.int64))
        assert TensorDataset(p1).hash() != TensorDataset(p2).hash()

    def test_different_shape_different_hash(self, tmp_path: Path) -> None:
        p1 = tmp_path / "d1.npz"
        p2 = tmp_path / "d2.npz"
        np.savez(p1, x=np.zeros((10, 3), dtype=np.float32), y=np.zeros(10, dtype=np.int64))
        np.savez(p2, x=np.zeros((10, 4), dtype=np.float32), y=np.zeros(10, dtype=np.int64))
        assert TensorDataset(p1).hash() != TensorDataset(p2).hash()

    def test_hash_cached(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz")
        d = TensorDataset(p)
        h1 = d.hash()
        h2 = d.hash()
        assert h1 == h2
        assert d._cached_hash == h1


class TestLoadFormats:
    def test_loads_npz_supervised(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz", n=50, d=3)
        d = TensorDataset(p)
        data = d.load()
        assert set(data.keys()) >= {"x", "y"}
        assert data["x"].shape == (50, 3)
        assert data["y"].shape == (50,)

    def test_loads_pt(self, tmp_path: Path) -> None:
        p = tmp_path / "d.pt"
        torch.save({"x": torch.zeros(10, 3), "y": torch.zeros(10, dtype=torch.long)}, p)
        data = TensorDataset(p).load()
        assert data["x"].shape == (10, 3)

    def test_rejects_unsupported_format(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="unsupported"):
            TensorDataset(p).load()

    def test_pt_rejects_non_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "d.pt"
        torch.save(torch.zeros(10), p)
        with pytest.raises(TypeError, match="expected dict"):
            TensorDataset(p).load()

    def test_pt_rejects_non_tensor_values(self, tmp_path: Path) -> None:
        p = tmp_path / "d.pt"
        torch.save({"x": torch.zeros(10), "y": [1, 2, 3]}, p)
        with pytest.raises(TypeError, match="expected Tensor"):
            TensorDataset(p).load()

    def test_schema_validation_on_load(self, tmp_path: Path) -> None:
        p = tmp_path / "d.npz"
        np.savez(p, x=np.zeros((10, 3), dtype=np.float32))  # no y
        with pytest.raises(SchemaError, match="missing required"):
            TensorDataset(p).load()

    def test_custom_schema_accepted(self, tmp_path: Path) -> None:
        p = tmp_path / "d.npz"
        np.savez(p, x=np.zeros((10, 3), dtype=np.float32))
        schema = DataSchema(name="x_only", required=frozenset({"x"}))
        TensorDataset(p, schema=schema).load()


class TestLen:
    def test_len_matches_leading_axis(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz", n=37)
        assert len(TensorDataset(p)) == 37


class TestSplit:
    def test_split_sizes(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz", n=100)
        d = TensorDataset(p)
        train, val, test = d.split([0.6, 0.2, 0.2], seed=0)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_sums_to_full(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz", n=97)  # not evenly divisible
        d = TensorDataset(p)
        parts = d.split([0.33, 0.33, 0.34], seed=0)
        assert sum(len(p) for p in parts) == 97

    def test_split_ratios_must_sum_to_one(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz")
        d = TensorDataset(p)
        with pytest.raises(ValueError, match="must sum to 1"):
            d.split([0.5, 0.3], seed=0)

    def test_split_is_deterministic(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz")
        d = TensorDataset(p)
        a1, b1 = d.split([0.5, 0.5], seed=42)
        a2, b2 = d.split([0.5, 0.5], seed=42)
        assert a1.hash() == a2.hash()
        assert b1.hash() == b2.hash()

    def test_split_seed_changes_partition(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz")
        d = TensorDataset(p)
        a1, _ = d.split([0.5, 0.5], seed=42)
        a2, _ = d.split([0.5, 0.5], seed=43)
        assert a1.hash() != a2.hash()

    def test_subset_hash_chains_from_parent(self, tmp_path: Path) -> None:
        p1 = tmp_path / "d1.npz"
        p2 = tmp_path / "d2.npz"
        np.savez(p1, x=np.zeros((20, 2), dtype=np.float32), y=np.zeros(20, dtype=np.int64))
        np.savez(p2, x=np.ones((20, 2), dtype=np.float32), y=np.zeros(20, dtype=np.int64))
        a, _ = TensorDataset(p1).split([0.5, 0.5], seed=0)
        c, _ = TensorDataset(p2).split([0.5, 0.5], seed=0)
        assert a.hash() != c.hash()

    def test_subset_load_returns_correct_size(self, tmp_path: Path) -> None:
        p = _make_npz(tmp_path / "d.npz", n=50)
        d = TensorDataset(p)
        train, val = d.split([0.8, 0.2], seed=0)
        data = train.load()
        assert data["x"].shape[0] == 40
        assert data["y"].shape[0] == 40


class TestDatasetABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Dataset()  # type: ignore[abstract]
