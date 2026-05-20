"""Unit tests for History: append, columnar view, save/load, pandas export."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perfsim.history import History


class TestAppend:
    def test_starts_empty(self) -> None:
        h = History()
        assert len(h) == 0

    def test_append_grows(self) -> None:
        h = History()
        h.append(round=0, theta=torch.zeros(3))
        h.append(round=1, theta=torch.ones(3))
        assert len(h) == 2

    def test_indexing(self) -> None:
        h = History()
        h.append(round=0, theta=torch.zeros(3))
        assert h[0]["round"] == 0
        assert torch.allclose(h[0]["theta"], torch.zeros(3))


class TestToDict:
    def test_empty(self) -> None:
        assert History().to_dict() == {}

    def test_stacks_uniform_tensors(self) -> None:
        h = History()
        for i in range(5):
            h.append(round=i, theta=torch.full((3,), float(i)))
        out = h.to_dict()
        assert out["round"] == [0, 1, 2, 3, 4]
        assert isinstance(out["theta"], torch.Tensor)
        assert out["theta"].shape == (5, 3)

    def test_returns_list_for_mixed_shapes(self) -> None:
        h = History()
        h.append(round=0, x=torch.zeros(3))
        h.append(round=1, x=torch.zeros(5))
        out = h.to_dict()
        assert isinstance(out["x"], list)
        assert len(out["x"]) == 2


class TestDataframe:
    def test_dataframe_columns(self) -> None:
        pd = pytest.importorskip("pandas")
        h = History()
        for i in range(3):
            h.append(round=i, theta=torch.full((2,), float(i)), loss=torch.tensor(float(i) ** 2))
        df = h.to_dataframe()
        assert list(df.columns) == ["round", "theta", "loss"]
        assert len(df) == 3
        assert df["round"].tolist() == [0, 1, 2]
        assert df["loss"].tolist() == [0.0, 1.0, 4.0]


class TestSaveLoad:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        h = History()
        h.append(round=0, theta=torch.tensor([1.0, 2.0]), name="r0")
        h.append(round=1, theta=torch.tensor([3.0, 4.0]), name="r1")
        p = tmp_path / "h.json"
        h.save(p)
        loaded = History.load(p)
        assert len(loaded) == 2
        assert loaded[0]["round"] == 0
        assert loaded[1]["name"] == "r1"
        assert loaded[0]["theta"] == [1.0, 2.0]

    def test_pt_round_trip(self, tmp_path: Path) -> None:
        h = History()
        h.append(round=0, theta=torch.tensor([1.0, 2.0]))
        h.append(round=1, theta=torch.tensor([3.0, 4.0]))
        p = tmp_path / "h.pt"
        h.save(p)
        loaded = History.load(p)
        assert len(loaded) == 2
        assert torch.allclose(loaded[0]["theta"], torch.tensor([1.0, 2.0]))

    def test_unsupported_save_suffix_raises(self, tmp_path: Path) -> None:
        h = History()
        with pytest.raises(ValueError, match="unsupported suffix"):
            h.save(tmp_path / "h.txt")

    def test_unsupported_load_suffix_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "h.txt"
        p.write_text("[]")
        with pytest.raises(ValueError, match="unsupported suffix"):
            History.load(p)
