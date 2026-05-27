"""KaggleDataset: download a Kaggle competition file (cached) as a TabularDataset."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence

import torch

from perfsim.core.types import SUPERVISED_SCHEMA, DataSchema
from perfsim.datasets.tabular import TabularDataset

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    _HAS_KAGGLE = True
except Exception:
    KaggleApi = None  # type: ignore[assignment,misc]
    _HAS_KAGGLE = False


def default_cache_dir() -> Path:
    """Default cache directory: `~/.cache/perfsim/datasets`."""
    return Path.home() / ".cache" / "perfsim" / "datasets"


class KaggleDataset(TabularDataset):
    """TabularDataset backed by a Kaggle competition download."""

    def __init__(
        self,
        competition: str,
        file: str,
        *,
        label_col: str,
        feature_cols: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        drop_na: bool = True,
        x_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        schema: DataSchema = SUPERVISED_SCHEMA,
    ) -> None:
        self._competition = competition
        self._file = file
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._comp_dir = self._cache_dir / competition
        self._target_path = self._comp_dir / file
        self._ensure_downloaded(force=force_download)
        super().__init__(
            path=self._target_path,
            label_col=label_col,
            feature_cols=feature_cols,
            drop_na=drop_na,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            schema=schema,
        )

    @property
    def competition(self) -> str:
        return self._competition

    @property
    def cache_path(self) -> Path:
        return self._target_path

    def _ensure_downloaded(self, *, force: bool) -> None:
        if self._target_path.exists() and not force:
            return
        self._comp_dir.mkdir(parents=True, exist_ok=True)
        self._download()
        if not self._target_path.exists():
            available = sorted(p.name for p in self._comp_dir.rglob("*") if p.is_file())
            raise FileNotFoundError(
                f"After downloading {self._competition!r}, expected file "
                f"{self._target_path} not found. Available in {self._comp_dir}: {available}"
            )

    def _download(self) -> None:
        """Download competition files into `self._comp_dir` and unzip any archives."""
        if KaggleApi is None:
            raise ImportError(
                "KaggleDataset requires the 'kaggle' extra. "
                "Install with: pip install 'perfsim[kaggle]'"
            )
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(
            self._competition, path=str(self._comp_dir), quiet=True
        )
        for zp in self._comp_dir.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(self._comp_dir)
            zp.unlink()
