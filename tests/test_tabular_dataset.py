"""Unit tests for TabularDataset (CSV / parquet)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perfsim.core import SchemaError
from perfsim.core.types import DataSchema
from perfsim.datasets import TabularDataset

pd = pytest.importorskip("pandas")


def _write_csv(tmp_path: Path, **kwargs) -> Path:
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


class TestLoad:
    def test_load_csv_explicit_features(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, feature_cols=["feat_a", "feat_b"], label_col="label")
        data = ds.load()
        assert data["x"].shape == (5, 2)
        assert data["y"].shape == (5,)
        assert data["x"].dtype == torch.float32
        assert ds.feature_columns == ["feat_a", "feat_b"]

    def test_load_csv_default_features(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, label_col="label")
        data = ds.load()
        assert data["x"].shape == (5, 2)
        assert ds.feature_columns == ["feat_a", "feat_b"]

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        with pytest.raises(ValueError, match="label_col"):
            TabularDataset(p, label_col="missing").load()

    def test_unknown_feature_raises(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, feature_cols=["feat_a", "missing"], label_col="label")
        with pytest.raises(ValueError, match="feature_cols missing"):
            ds.load()

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="unsupported suffix"):
            TabularDataset(p, label_col="label").load()

    def test_drop_na_default(self, tmp_path: Path) -> None:
        p = tmp_path / "with_na.csv"
        pd.DataFrame(
            {"feat_a": [1.0, None, 3.0], "feat_b": [0.1, 0.2, 0.3], "label": [0, 1, 0]}
        ).to_csv(p, index=False)
        ds = TabularDataset(p, label_col="label")
        data = ds.load()
        assert data["x"].shape == (2, 2)

    def test_drop_na_off(self, tmp_path: Path) -> None:
        p = tmp_path / "with_na.csv"
        pd.DataFrame(
            {"feat_a": [1.0, None, 3.0], "feat_b": [0.1, 0.2, 0.3], "label": [0, 1, 0]}
        ).to_csv(p, index=False)
        ds = TabularDataset(p, label_col="label", drop_na=False)
        data = ds.load()
        assert data["x"].shape == (3, 2)

    def test_custom_y_dtype_long(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, label_col="label", y_dtype=torch.long)
        data = ds.load()
        assert data["y"].dtype == torch.long


class TestParquet:
    def test_parquet_round_trip(self, tmp_path: Path) -> None:
        pq = pytest.importorskip("pyarrow")
        p = tmp_path / "data.parquet"
        pd.DataFrame(
            {"feat_a": [1.0, 2.0], "feat_b": [0.1, 0.2], "label": [0, 1]}
        ).to_parquet(p)
        ds = TabularDataset(p, label_col="label")
        data = ds.load()
        assert data["x"].shape == (2, 2)


class TestSchemaPropagation:
    def test_loaded_data_satisfies_supervised_schema(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, label_col="label")
        ds.load()  # should not raise SchemaError

    def test_custom_schema_requirement_violated(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        picky = DataSchema(name="picky", required=frozenset({"x", "y", "weight"}))
        ds = TabularDataset(p, label_col="label", schema=picky)
        with pytest.raises(SchemaError, match="missing required"):
            ds.load()


class TestHashAndSplit:
    def test_hash_stable_across_instances(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        a = TabularDataset(p, label_col="label")
        b = TabularDataset(p, label_col="label")
        assert a.hash() == b.hash()

    def test_hash_differs_with_different_features(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        a = TabularDataset(p, feature_cols=["feat_a"], label_col="label")
        b = TabularDataset(p, feature_cols=["feat_b"], label_col="label")
        assert a.hash() != b.hash()

    def test_split_works(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path)
        ds = TabularDataset(p, label_col="label")
        train, val = ds.split([0.6, 0.4], seed=0)
        assert len(train) == 3
        assert len(val) == 2
