"""TensorDataset and InMemoryDataset: load tensors from disk or memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from perfsim.core.dataset import Dataset
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class TensorDataset(Dataset):
    """Dataset backed by tensors on disk (.npz / .pt / .pth)."""

    def __init__(
        self,
        path: str | Path,
        schema: DataSchema = SUPERVISED_SCHEMA,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self.produces_schema = schema

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> Data:
        suffix = self._path.suffix.lower()
        if suffix == ".npz":
            with np.load(self._path) as npz:
                return {k: torch.from_numpy(np.asarray(npz[k])) for k in npz.files}
        if suffix in (".pt", ".pth"):
            obj = torch.load(self._path, map_location="cpu", weights_only=True)
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{self._path}: torch.load returned {type(obj).__name__}; "
                    f"expected dict[str, Tensor]"
                )
            for k, v in obj.items():
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"{self._path}: key {k!r} is {type(v).__name__}; "
                        f"expected Tensor"
                    )
            return obj
        raise ValueError(
            f"{self._path}: unsupported suffix {suffix!r}; expected .npz, .pt, or .pth"
        )


class InMemoryDataset(Dataset):
    """Dataset backed by an in-memory data dict."""

    def __init__(
        self,
        data: Data,
        *,
        schema: DataSchema = SUPERVISED_SCHEMA,
    ) -> None:
        super().__init__()
        self._data = {k: v.detach().clone() for k, v in data.items()}
        self.produces_schema = schema

    def _load(self) -> Data:
        return self._data
