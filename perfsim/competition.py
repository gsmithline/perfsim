"""Round loop for P platforms retraining against one shared world."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from perfsim.core.learner import Learner
from perfsim.core.types import Data


def run_competition(
    env,
    learners: Sequence[Learner],
    *,
    n_rounds: int,
    epoch_size: int = 1,
    seed: int = 0,
    on_round: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Simulator.run's loop fanned over platforms: round 0 deploys the untrained
    models; each round trains each learner on the data its platform induced."""
    env.reset(seed=seed)
    prev: list[Data | None] = [None] * len(learners)
    history: list[dict[str, Any]] = []
    for t in range(n_rounds):
        for learner, data in zip(learners, prev):
            if data is not None:
                learner.train(data)
        prev = env.run([lrn.model for lrn in learners], n_steps=epoch_size)
        record: dict[str, Any] = {"round": t}
        if on_round is not None:
            on_round(t, record)
        history.append(record)
    return history
