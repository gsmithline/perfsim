"""Concrete Dataset loaders.

- `TensorDataset` (v0): loads from .npz, .pt, .pth.
- `TabularDataset` (v1): loads CSV/parquet via pandas.
- `KaggleDataset` (v1): downloads a Kaggle competition file and wraps it as
  a TabularDataset; requires perfsim[kaggle] and Kaggle credentials.
"""

from perfsim.datasets.kaggle import KaggleDataset, default_cache_dir
from perfsim.datasets.tabular import TabularDataset
from perfsim.datasets.tensor import InMemoryDataset, TensorDataset

__all__ = [
    "InMemoryDataset",
    "KaggleDataset",
    "TabularDataset",
    "TensorDataset",
    "default_cache_dir",
]
