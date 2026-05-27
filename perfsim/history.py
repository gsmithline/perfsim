"""History: per-round records as a list of dicts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from torch import Tensor


class History:
    """Per-round records buffered as dicts of (tensor | scalar | str)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def append(self, **record: Any) -> None:
        """Append a round's record. Tensors are kept as tensors."""
        self._records.append(dict(record))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def to_dict(self) -> dict[str, Any]:
        """Columnar view: stacks uniform-shape tensor columns, lists the rest."""
        if not self._records:
            return {}
        keys = list(self._records[0].keys())
        out: dict[str, Any] = {}
        for k in keys:
            values = [r.get(k) for r in self._records]
            if all(isinstance(v, Tensor) for v in values):
                cpu_values = [v.detach().cpu() for v in values]
                shapes = {tuple(v.shape) for v in cpu_values}
                if len(shapes) == 1:
                    out[k] = torch.stack(cpu_values)
                else:
                    out[k] = cpu_values
            else:
                out[k] = values
        return out

    def to_dataframe(self):
        """Pandas DataFrame view. Requires perfsim[tabular]."""
           
        rows = []
        for r in self._records:
            row: dict[str, Any] = {}
            for k, v in r.items():
                if isinstance(v, Tensor):
                    if v.ndim == 0:
                        row[k] = v.item()
                    else:
                        row[k] = v.detach().cpu().tolist()
                else:
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Save to .pt (torch.save) or .json. Tensors converted to lists for JSON."""
        p = Path(path)
        if p.suffix == ".pt":
            torch.save(self._records, p)
            return
        if p.suffix == ".json":
            rows = []
            for r in self._records:
                row: dict[str, Any] = {}
                for k, v in r.items():
                    if isinstance(v, Tensor):
                        row[k] = v.detach().cpu().tolist()
                    else:
                        row[k] = v
                rows.append(row)
            p.write_text(json.dumps(rows, indent=2))
            return
        raise ValueError(f"unsupported suffix {p.suffix!r}; expected .pt or .json")

    @classmethod
    def load(cls, path: str | Path) -> "History":
        p = Path(path)
        h = cls()
        if p.suffix == ".pt":
            records = torch.load(p, weights_only=False)
            if not isinstance(records, list):
                raise TypeError(
                    f"{p}: expected list of dicts, got {type(records).__name__}"
                )
            h._records = records
            return h
        if p.suffix == ".json":
            h._records = json.loads(p.read_text())
            return h
        raise ValueError(f"unsupported suffix {p.suffix!r}; expected .pt or .json")
