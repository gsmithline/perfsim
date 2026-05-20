"""TabularDataset: loads CSV or parquet via pandas into a data dict.

Specify `label_col`; `feature_cols` defaults to "all other columns" if not
passed. Output dict: `{"x": Tensor[N, D], "y": Tensor[N]}`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from perfsim.core.dataset import Dataset
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
import pandas as pd


class TabularDataset(Dataset):
    """CSV / parquet -> supervised data dict.

    Args:
        path: file path with suffix .csv, .parquet, or .pq.
        label_col: name of the label column.
        feature_cols: names of feature columns; if None, all columns except
            `label_col` are used.
        drop_na: drop rows with NaN in any used column. Default True.
        x_dtype, y_dtype: tensor dtypes for x and y.
        schema: declared output schema; default supervised (x, y).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        label_col: str,
        feature_cols: Sequence[str] | None = None,
        drop_na: bool = True,
        x_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        schema: DataSchema = SUPERVISED_SCHEMA,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._label_col = label_col
        self._feature_cols: list[str] | None = (
            list(feature_cols) if feature_cols is not None else None
        )
        self._drop_na = drop_na
        self._x_dtype = x_dtype
        self._y_dtype = y_dtype
        self.produces_schema = schema
        self._resolved_feature_cols: list[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def feature_columns(self) -> list[str] | None:
        """Resolved feature column names after first load; None until then."""
        return self._resolved_feature_cols

    def _read_df(self):
        """
        add support for .pt 
        """
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self._path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(self._path)
        raise ValueError(
            f"{self._path}: unsupported suffix {suffix!r}; expected .csv, .parquet, or .pq"
        )

    def _load(self) -> Data:
        df = self._read_df()
        if self._label_col not in df.columns:
            raise ValueError(
                f"label_col {self._label_col!r} not in dataframe columns {list(df.columns)}"
            )
        if self._feature_cols is None:
            feature_cols = [c for c in df.columns if c != self._label_col]
        else:
            feature_cols = self._feature_cols
            missing = set(feature_cols) - set(df.columns)
            if missing:
                raise ValueError(
                    f"feature_cols missing from dataframe: {sorted(missing)}; "
                    f"have {list(df.columns)}"
                )
        used = [*feature_cols, self._label_col]
        if self._drop_na:
            df = df.dropna(subset=used)
        x = torch.tensor(df[feature_cols].to_numpy(), dtype=self._x_dtype)
        y = torch.tensor(df[self._label_col].to_numpy(), dtype=self._y_dtype)
        self._resolved_feature_cols = list(feature_cols)
        return {"x": x, "y": y}
