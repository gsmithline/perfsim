"""Shared types: Data, schemas, ConfigBase, AgentHandle.

The supervised schema (`x`, `y`) is exercised in v1; the trajectory schema is a
v2 placeholder, designed alongside the first concrete RL Learner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, TypeAlias

from torch import Tensor

Data: TypeAlias = dict[str, Tensor]


class SchemaError(Exception):
    """Raised when a Data dict does not satisfy a DataSchema."""


@dataclass(frozen=True)
class DataSchema:
    """Declares the field names a Learner expects or a World produces.

    Field names are checked at binding time. Tensor shapes and dtypes are not
    enforced here; those are component-internal concerns.
    """

    name: str
    required: frozenset[str] = field(default_factory=frozenset)
    optional: frozenset[str] = field(default_factory=frozenset)

    def validate(self, data: Mapping[str, Tensor]) -> None:
        missing = self.required - data.keys()
        if missing:
            raise SchemaError(
                f"Schema {self.name!r}: missing required fields {sorted(missing)}; "
                f"got {sorted(data.keys())}"
            )

    def covers(self, other: "DataSchema") -> bool:
        """True if this schema (produced by a World) satisfies what `other`
        (required by a Learner) needs.

        Used by Simulator.bind(): a World's produces_schema must cover the
        Learner's accepted_schema. `other.required` must be a subset of
        `self.required | self.optional`.
        """
        return other.required <= (self.required | self.optional)


SUPERVISED_SCHEMA: DataSchema = DataSchema(
    name="supervised",
    required=frozenset({"x", "y"}),
)

TRAJECTORY_SCHEMA: DataSchema = DataSchema( #FOR RL ROLLOUTS WHEN WE GET THERE 
    name="trajectory",
    required=frozenset(),
    optional=frozenset(),
)
"""v2 placeholder. Marked empty so any Learner declaring it as accepted today
fails binding loudly until v2 designs the concrete fields alongside the first
concrete RL Learner."""


@dataclass(frozen=True)
class ConfigBase:
    """Base class for run configs. Provides JSON ser/deser and content hash."""

    def to_dict(self) -> dict[str, Any]:
        return _asdict_recursive(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default)

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


def _asdict_recursive(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _asdict_recursive(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    if isinstance(obj, tuple):
        return [_asdict_recursive(v) for v in obj]
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    return str(obj)


@dataclass(frozen=True)
class AgentHandle:
    """Reference to an agent. Used by Executors to dispatch skill calls."""

    id: str
    role: str
    endpoint: str | None = None
