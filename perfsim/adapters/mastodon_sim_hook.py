"""Global registry shared between mastodon-sim's scene.py and the perfsim
MastodonSimEnvironment adapter, so neither imports the other.

perfsim calls register_ranker / register_post_pool before an episode and the
clear_* functions after; scene.py reads via get_active_ranker / get_post_pool
and falls back to its own behavior when they return None (so mastodon-sim still
runs standalone). The ranker exposes rank(viewer, pool, k) -> list[dict] where
each dict carries the keys scene.py reads (content, media_attachments,
account.display_name, account.username, id).

An RLock guards mutation since the gamemaster runs agents in a ThreadPoolExecutor.
HFCausalLMModel.forward is not thread-safe, so prefer serial-agent runs.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Protocol


class Ranker(Protocol):
    """Duck-typed contract for objects passed to `register_ranker`."""

    def rank(self, viewer: str, pool: Any, k: int) -> list[dict]:
        ...


_lock = threading.RLock()
_active_ranker: Optional[Ranker] = None
_post_pool: Any = None


def register_ranker(ranker: Ranker) -> None:
    """Install `ranker` as the active recommender, replacing any prior one."""
    global _active_ranker
    with _lock:
        _active_ranker = ranker


def clear_ranker() -> None:
    """Remove the active recommender (get_active_ranker becomes None)."""
    global _active_ranker
    with _lock:
        _active_ranker = None


def get_active_ranker() -> Optional[Ranker]:
    """Return the registered ranker, or None (scene.py's no-perfsim fallback)."""
    return _active_ranker


def register_post_pool(pool: Any) -> None:
    """Install `pool` as the opaque candidate pool for the ranker."""
    global _post_pool
    with _lock:
        _post_pool = pool


def clear_post_pool() -> None:
    """Remove the candidate pool."""
    global _post_pool
    with _lock:
        _post_pool = None


def get_post_pool() -> Any:
    """Return the registered pool or `None`."""
    return _post_pool
