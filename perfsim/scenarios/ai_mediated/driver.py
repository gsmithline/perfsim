"""Retrain-regime driver for the AI-mediated loop.

The base Simulator only does "replace" (train on the last round). The interesting
contrasts in this project live in HOW the platform composes its training set
across rounds (Exp 5):

- replace:         train on the newest mediated data only.
- accumulate:      train on the union of all mediated data so far.
- clean_anchor:    train on alpha*(clean raw human data) + (1-alpha)*(newest).
- mediated_anchor: train on alpha*(round-0 mediated data) + (1-alpha)*(newest).
                   The "human" anchor is itself already AI-mediated.

The clean_anchor-vs-mediated_anchor contrast is the empirical form of "post-
deployment human data is already AI-mediated": clean anchor is the textbook
collapse defense; mediated anchor is what you actually have in a deployed world.

This driver owns the buffer and composition; the env (MediationWorld) stays a
one-shot D(theta). Mixing samples whole rows to a fixed `total_size` so every
regime trains on the same volume (no volume confound).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from perfsim.core.predictor import Predictor
from perfsim.core.types import Data
from perfsim.environments.mediation import MediationWorld

REGIMES = ("replace", "accumulate", "clean_anchor", "mediated_anchor")

ProbeFn = Callable[[Data], float]


def _concat(chunks: list[Data]) -> Data:
    keys = chunks[0].keys()
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def _sample_rows(data: Data, n: int, generator: torch.Generator) -> Data:
    if n <= 0:
        return {k: v[:0].clone() for k, v in data.items()}
    avail = next(iter(data.values())).shape[0]
    replace = n > avail
    if replace:
        idx = torch.randint(avail, (n,), generator=generator)
    else:
        idx = torch.randperm(avail, generator=generator)[:n]
    return {k: v[idx].clone() for k, v in data.items()}


def _mix(
    anchor: Data,
    recent: Data,
    *,
    alpha: float,
    total: int,
    generator: torch.Generator,
) -> Data:
    n_anchor = int(round(alpha * total))
    n_recent = total - n_anchor
    return _concat([
        _sample_rows(anchor, n_anchor, generator),
        _sample_rows(recent, n_recent, generator),
    ])


def _compose(
    regime: str,
    buffer: list[Data],
    *,
    clean_anchor: Data,
    alpha: float,
    total: int,
    generator: torch.Generator,
) -> Data:
    recent = buffer[-1]
    if regime == "replace":
        return recent
    if regime == "accumulate":
        return _concat(buffer)
    if regime == "clean_anchor":
        return _mix(clean_anchor, recent, alpha=alpha, total=total, generator=generator)
    if regime == "mediated_anchor":
        return _mix(buffer[0], recent, alpha=alpha, total=total, generator=generator)
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")


def run_mediated(
    world: MediationWorld,
    predictor: Predictor,
    *,
    n_rounds: int,
    regime: str,
    initial_data: Data | None = None,
    seed: int = 0,
    alpha: float = 0.5,
    clean_anchor: Data | None = None,
    total_size: int | None = None,
    probes: Optional[dict[str, ProbeFn]] = None,
    on_round: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the AI-mediated performative loop and return per-round probe records.

    Round shape (mirrors Simulator): train on the composed prev set, deploy,
    mediate the fixed population under the new theta, buffer it, probe it.
    `initial_data` (the raw clean data) trains theta_0; defaults to world.raw_data.
    For clean_anchor, `clean_anchor` defaults to world.raw_data.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1; got {n_rounds}")
    probes = probes or {}
    total = total_size if total_size is not None else world.n_agents
    if initial_data is None:
        initial_data = world.raw_data
    if clean_anchor is None:
        clean_anchor = world.raw_data

    world.reset(seed=seed)
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    buffer: list[Data] = []
    prev: Data | None = initial_data
    records: list[dict[str, Any]] = []
    for t in range(n_rounds):
        if prev is not None:
            predictor.train(prev)
        handle = predictor.deploy()
        data = world.run(handle, n_steps=1)
        buffer.append({k: v.clone() for k, v in data.items()})
        record: dict[str, Any] = {"round": t}
        for name, fn in probes.items():
            record[name] = fn(data)
        records.append(record)
        if on_round is not None:
            on_round(t, record)
        prev = _compose(
            regime,
            buffer,
            clean_anchor=clean_anchor,
            alpha=alpha,
            total=total,
            generator=gen,
        )
    return records
