"""Dataset ABC: load, hash, split."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor

from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


class Dataset(ABC):
    """Base class for Datasets."""

    produces_schema: DataSchema = SUPERVISED_SCHEMA

    def __init__(self) -> None:
        self._cached_data: Data | None = None
        self._cached_hash: str | None = None

    @abstractmethod
    def _load(self) -> Data:
        """Produce the data dict. Called once; result is cached by `load()`."""

    def load(self) -> Data:
        if self._cached_data is None:
            data = self._load()
            self.produces_schema.validate(data)
            self._cached_data = data
        return self._cached_data

    def hash(self) -> str:
        if self._cached_hash is None:
            self._cached_hash = self._compute_hash()
        return self._cached_hash

    def _compute_hash(self) -> str:
        data = self.load()
        h = hashlib.sha256()
        h.update(self.produces_schema.name.encode("utf-8"))
        for key in sorted(data.keys()):
            t = data[key].detach().cpu().contiguous()
            h.update(key.encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        return h.hexdigest()[:32]

    def __len__(self) -> int:
        data = self.load()
        if not data:
            return 0
        first = next(iter(data.values()))
        return int(first.shape[0])

    def split(self, ratios: Sequence[float], seed: int) -> tuple["Dataset", ...]:
        """Split into subsets along the leading axis with the given ratios.

        Returns SubsetDataset views; subset hashes chain from this Dataset's
        hash, the index tensor, and the seed.
        """
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {sum(ratios)}")
        n = len(self)
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        sizes = [int(n * r) for r in ratios]
        sizes[-1] = n - sum(sizes[:-1])
        subsets: list[Dataset] = []
        start = 0
        for size in sizes:
            idx = perm[start : start + size]
            subsets.append(SubsetDataset(self, idx, seed=seed))
            start += size
        return tuple(subsets)


class SubsetDataset(Dataset):
    """View into a parent Dataset along the leading axis."""

    def __init__(self, parent: Dataset, indices: Tensor, seed: int) -> None:
        super().__init__()
        self._parent = parent
        self._indices = indices.detach().clone()
        self._seed = seed
        self.produces_schema = parent.produces_schema

    def _load(self) -> Data:
        parent_data = self._parent.load()
        return {k: v[self._indices] for k, v in parent_data.items()}

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"subset:")
        h.update(self._parent.hash().encode("utf-8"))
        h.update(self._indices.numpy().tobytes())
        h.update(str(self._seed).encode("utf-8"))
        return h.hexdigest()[:32]
