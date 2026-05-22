"""KaggleDataset: 

download a Kaggle competition file (cached) and expose it
as a `TabularDataset`.

Caches to `~/.cache/perfsim/datasets/{competition}/{file}` by default. On
first use, downloads via the Kaggle API; subsequent uses read from the
cache. Requires `pip install perfsim[kaggle]` and Kaggle CLI credentials at
`~/.kaggle/kaggle.json`.

The hash inherited from `TabularDataset` is over the loaded tensors, so it
is stable as long as the cached file content is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from perfsim.core.types import SUPERVISED_SCHEMA, DataSchema
from perfsim.datasets.tabular import TabularDataset
import zipfile

# kaggle import is intentionally deferred to call-site (see _download).
# Importing it at module load triggers the kaggle CLI's auto-auth check,
# which fails on machines without ~/.kaggle/kaggle.json even when
# KaggleDataset is never instantiated. Transitive importers (any code that
# touches perfsim.datasets via the perfsim.scenarios __init__) would
# otherwise fail at import on cluster nodes without Kaggle credentials.


def default_cache_dir() -> Path:
    """Default cache directory: `~/.cache/perfsim/datasets`."""
    return Path.home() / ".cache" / "perfsim" / "datasets"


class KaggleDataset(TabularDataset):
    """A `TabularDataset` whose file is downloaded from a Kaggle competition.

    Args:
        competition: Kaggle competition slug (e.g. ``"GiveMeSomeCredit"``).
        file: filename to use from the competition zip (e.g. ``"cs-training.csv"``).
        label_col: name of the label column.
        feature_cols: feature column names; None for all-other.
        cache_dir: where to cache downloads. Default: `~/.cache/perfsim/datasets`.
        force_download: re-download even if the cached file exists.
        Other kwargs are forwarded to `TabularDataset`.
    """

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

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise ImportError(
                "KaggleDataset requires the 'kaggle' extra. "
                "Install with: pip install 'perfsim[kaggle]'"
            ) from exc

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(
            self._competition, path=str(self._comp_dir), quiet=True
        )
        for zp in self._comp_dir.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(self._comp_dir)
            zp.unlink()
