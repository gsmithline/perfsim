"""Tiny global registry shared between mastodon-sim's `scene.py` and the
perfsim `MastodonSimEnvironment` adapter.

Why this exists
---------------
mastodon-sim's phone-scene inner loop fetches the viewer's timeline via a
single concrete call (see `scene.py` lines ~162 and ~267):

    timeline = mastodon_ops.get_own_timeline(p_username, limit=10)

Stage 1 already wrapped these in `if app.perform_operations: ... else: []`
fallbacks. To insert a learned recommender we want a way for perfsim to
"register" a ranker before each mastodon-sim episode runs, scene.py to pick
it up if present, and perfsim to clear it afterwards. This module is that
shared state - no import of perfsim from mastodon-sim and vice versa.

Contract
--------
- `register_ranker(ranker)`: called by perfsim before kicking off an episode.
  `ranker` must expose a `.rank(viewer, pool, k) -> list[dict]` method
  returning a list of "post-shaped" dicts (see Post Schema below).
- `register_post_pool(pool)`: called by perfsim to provide the candidate
  pool the ranker should score over. `pool` is just an opaque object the
  ranker understands; scene.py only reads it via `get_post_pool()` to hand
  it back to the ranker.
- `clear_ranker()` / `clear_post_pool()`: called by perfsim after the
  episode to leave global state clean. If the adapter crashes mid-episode,
  the next adapter run also re-registers, so a stuck ranker won't poison a
  subsequent fresh `MastodonSimEnvironment`.
- `get_active_ranker()` / `get_post_pool()`: called by scene.py. Returns
  `None` when nothing is registered - in that case scene.py keeps its old
  `timeline = []` fallback. This is what preserves the "mastodon-sim still
  runs standalone without perfsim" property.

Post schema (what scene.py expects from `ranker.rank`)
------------------------------------------------------
Each entry must be a dict with at minimum the keys scene.py reads
(`content`, `media_attachments`, `account.display_name`,
`account.username`, `id`). The recommender wrapper in
`perfsim.adapters.mastodon_sim` builds these dicts directly so the
ranker side does not need to know about the Mastodon HTTP API. See
`mastodon_sim.MastodonSimEnvironment._build_recommender` for the
canonical builder.

Thread safety
-------------
mastodon-sim's gamemaster runs agents in a `ThreadPoolExecutor`, so
multiple scene.py threads can call `get_active_ranker()` concurrently.
A `threading.RLock` guards mutation. Reads are atomic by Python ref
semantics. The ranker itself must be thread-safe if you exercise the
parallel gamemaster - perfsim's recommender wrapper serializes LM calls
internally (HFCausalLMModel.forward is not thread-safe), so for now
prefer serial-agent runs by setting low Poisson activation rates or
patching the gamemaster to use a single-thread executor.
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
    """Install `ranker` as the active recommender. Replaces any prior
    registration without warning - the perfsim adapter is the sole caller
    and registers exactly once per episode.
    """
    global _active_ranker
    with _lock:
        _active_ranker = ranker


def clear_ranker() -> None:
    """Remove the active recommender. After this returns,
    `get_active_ranker()` is `None` until a new `register_ranker` call.
    """
    global _active_ranker
    with _lock:
        _active_ranker = None


def get_active_ranker() -> Optional[Ranker]:
    """Return the currently registered ranker, or `None` if there is none.

    scene.py uses this to decide whether to call the recommender or fall
    back to its existing `timeline = []` behavior. Designed so that
    importing this module is safe even when perfsim is not installed -
    `None` here means "perfsim not driving this run".
    """
    return _active_ranker


def register_post_pool(pool: Any) -> None:
    """Install `pool` as the candidate pool. The pool object is opaque -
    only the ranker that consumes it needs to know its shape.
    """
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
