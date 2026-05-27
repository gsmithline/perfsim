"""Unit tests for KaggleDataset.

We avoid live Kaggle calls in CI. Tests pre-populate the cache directory
or monkeypatch `_download` to simulate the download flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perfsim.datasets import KaggleDataset, TabularDataset, default_cache_dir

pd = pytest.importorskip("pandas")


def _write_competition_file(cache_dir: Path, comp: str, file: str) -> Path:
    comp_dir = cache_dir / comp
    comp_dir.mkdir(parents=True, exist_ok=True)
    p = comp_dir / file
    pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [0.1, 0.2, 0.3, 0.4],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(p, index=False)
    return p


class TestCacheHit:
    def test_skips_download_when_file_exists(self, tmp_path: Path) -> None:
        _write_competition_file(tmp_path, "TestComp", "data.csv")

        download_calls = {"n": 0}

        def fake_download(self) -> None:
            download_calls["n"] += 1

        original = KaggleDataset._download
        try:
            KaggleDataset._download = fake_download  # type: ignore[method-assign]
            ds = KaggleDataset(
                competition="TestComp",
                file="data.csv",
                label_col="label",
                cache_dir=tmp_path,
            )
            data = ds.load()
            assert download_calls["n"] == 0
            assert data["x"].shape == (4, 2)
            assert data["y"].shape == (4,)
        finally:
            KaggleDataset._download = original  # type: ignore[method-assign]

    def test_uses_custom_cache_dir(self, tmp_path: Path) -> None:
        _write_competition_file(tmp_path, "TestComp", "data.csv")
        ds = KaggleDataset(
            competition="TestComp",
            file="data.csv",
            label_col="label",
            cache_dir=tmp_path,
        )
        assert ds.cache_path == tmp_path / "TestComp" / "data.csv"
        assert ds.competition == "TestComp"


class TestDownloadFlow:
    def test_download_invoked_when_missing_and_creates_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        download_calls = {"n": 0}

        def fake_download(self) -> None:
            download_calls["n"] += 1
            # Simulate download by writing the expected file
            _write_competition_file(tmp_path, self._competition, self._file)

        monkeypatch.setattr(KaggleDataset, "_download", fake_download)

        ds = KaggleDataset(
            competition="TestComp",
            file="data.csv",
            label_col="label",
            cache_dir=tmp_path,
        )
        assert download_calls["n"] == 1
        assert (tmp_path / "TestComp" / "data.csv").exists()
        data = ds.load()
        assert data["x"].shape == (4, 2)

    def test_force_download_invokes_even_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_competition_file(tmp_path, "TestComp", "data.csv")

        download_calls = {"n": 0}

        def fake_download(self) -> None:
            download_calls["n"] += 1
            _write_competition_file(tmp_path, self._competition, self._file)

        monkeypatch.setattr(KaggleDataset, "_download", fake_download)

        KaggleDataset(
            competition="TestComp",
            file="data.csv",
            label_col="label",
            cache_dir=tmp_path,
            force_download=True,
        )
        assert download_calls["n"] == 1

    def test_raises_when_target_still_missing_after_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(self) -> None:
            return  # download succeeds but produces no expected file

        monkeypatch.setattr(KaggleDataset, "_download", fake_download)

        with pytest.raises(FileNotFoundError, match="not found"):
            KaggleDataset(
                competition="TestComp",
                file="data.csv",
                label_col="label",
                cache_dir=tmp_path,
            )


class TestKaggleImportError:
    def test_missing_kaggle_library_raises_helpful_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked_import(name: str, *args, **kwargs):
            if name.startswith("kaggle"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        # File missing -> _download invoked -> should raise ImportError
        with pytest.raises(ImportError, match="perfsim\\[kaggle\\]"):
            KaggleDataset(
                competition="NotARealComp",
                file="data.csv",
                label_col="label",
                cache_dir=tmp_path,
            )


class TestDefaultCacheDir:
    def test_default_path_under_home(self) -> None:
        d = default_cache_dir()
        assert d == Path.home() / ".cache" / "perfsim" / "datasets"


class TestHashStable:
    def test_kaggle_hash_matches_equivalent_tabular(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KaggleDataset.hash should be content-based (same as TabularDataset
        on the same file).
        """
        _write_competition_file(tmp_path, "TestComp", "data.csv")

        def no_download(self) -> None:
            return

        monkeypatch.setattr(KaggleDataset, "_download", no_download)

        kaggle_ds = KaggleDataset(
            competition="TestComp",
            file="data.csv",
            label_col="label",
            cache_dir=tmp_path,
        )
        tabular_ds = TabularDataset(
            tmp_path / "TestComp" / "data.csv", label_col="label"
        )
        assert kaggle_ds.hash() == tabular_ds.hash()
