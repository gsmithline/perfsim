from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from perfsim.core.dataset import Dataset
from perfsim.core.environment import Environment
from perfsim.core.learner import Learner
from perfsim.core.loss import Loss
from perfsim.core.predictor import Predictor
from perfsim.core.types import SchemaError
from perfsim.history import History
from perfsim.metrics import stability_gap


MetricFn = Callable[["Simulator"], Any]


class Simulator:
    """Epoch-loop orchestration."""

    def __init__(
        self,
        world: Environment | None = None,
        learner: Learner | None = None,
        loss: Loss | None = None,
        *,
        env: Environment | None = None,
        predictor: Predictor | None = None,
        metrics: Optional[dict[str, MetricFn]] = None,
        history: Optional[History] = None,
        dataset: Optional[Dataset] = None,
    ) -> None:
        env_arg = env if env is not None else world
        if env_arg is None:
            raise TypeError("Simulator requires `env=` (or legacy positional `world=`)")
        if predictor is None:
            if learner is None or loss is None:
                raise TypeError(
                    "Simulator: pass either `predictor=` or both `learner=` and `loss=`"
                )
            predictor = Predictor(model=learner.model, loss=loss, learner=learner)
        elif learner is not None or loss is not None:
            raise TypeError(
                "Simulator: pass either `predictor=` or `(learner=, loss=)`, not both"
            )

        self.env: Environment = env_arg
        self.predictor: Predictor = predictor
        self.metrics: dict[str, MetricFn] = metrics or {}
        self.history = history or History()
        self.dataset = dataset
        self._prev_theta: Tensor | None = None
        self._current_round: int = -1
        self._bind()

    # ---- Backward-compatible properties ------------------------------------
    # These let existing code (tests, scenarios) continue to read sim.world,
    # sim.learner, sim.loss without modification.

    @property
    def world(self) -> Environment:
        return self.env

    @property
    def learner(self) -> Learner:
        return self.predictor.learner

    @property
    def loss(self) -> Loss:
        return self.predictor.loss

    # ---- Binding and validation --------------------------------------------

    def _bind(self) -> None:
        """Validate the Environment's produced schema is accepted by the Learner."""
        produces = self.env.produces_schema
        learner = self.predictor.learner
        if not type(learner).accepts(produces):
            raise SchemaError(
                f"Binding error: Learner {type(learner).__name__} does not "
                f"accept Environment's schema {produces.name!r}. Learner accepts: "
                f"{[s.name for s in learner.accepted_schemas]}."
            )

    def _validate_epoch_size(self, epoch_size: int) -> None:
        if not isinstance(epoch_size, int) or epoch_size < 1:
            raise ValueError(f"epoch_size must be a positive int; got {epoch_size!r}")
        max_size = getattr(self.env, "max_meaningful_epoch_size", math.inf)
        if epoch_size > max_size:
            raise ValueError(
                f"epoch_size={epoch_size} exceeds {type(self.env).__name__}."
                f"max_meaningful_epoch_size={max_size}. This Environment's inner "
                f"step is not meaningful for N>1 under fixed theta."
            )

    @property
    def current_round(self) -> int:
        return self._current_round

    def run(
        self,
        n_rounds: int,
        *,
        epoch_size: int = 1,
        seed: int = 0,
        initial_data: dict[str, Tensor] | None = None,
        train_mask: Tensor | None = None,
        on_round: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> History:
        """Run the PP loop for `n_rounds` epochs of `epoch_size` env steps each.

        Per-round shape (Algorithm 1):

            prev = filter(initial_data, train_mask)
            for t in 0..n_rounds-1:
                if prev is not None: predictor.train(prev)
                final_data = env.run(predictor.deploy(), epoch_size)
                record(t)
                prev = filter(final_data, train_mask)

        initial_data=None skips round-0 training (theta_0 drives the first run).
        train_mask: optional (N,) bool selecting which env-produced rows feed
        training; masked-out rows still participate in dynamics but not training.
        """
        self._validate_epoch_size(epoch_size)
        if train_mask is not None:
            self._validate_train_mask(train_mask)
        self.env.reset(seed=seed)
        self._prev_theta = None
        self._current_round = -1
        prev_data: dict[str, Tensor] | None = self._mask_data(initial_data, train_mask)
        for t in range(n_rounds):
            self._current_round = t
            if prev_data is not None:
                self.predictor.train(prev_data)
            handle = self.predictor.deploy()
            final_data = self.env.run(handle, n_steps=epoch_size)
            self._record_round(t)
            if on_round is not None:
                on_round(t, self.history[-1])
            prev_data = self._mask_data(final_data, train_mask)
        return self.history

    @staticmethod
    def _validate_train_mask(mask: Tensor) -> None:
        if not isinstance(mask, Tensor):
            raise TypeError(
                f"train_mask must be a torch.Tensor; got {type(mask).__name__}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(
                f"train_mask must be a bool tensor; got dtype {mask.dtype}"
            )
        if mask.ndim != 1:
            raise ValueError(
                f"train_mask must be 1-D; got shape {tuple(mask.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError(
                "train_mask selects zero rows; predictor would have no "
                "training data"
            )

    @staticmethod
    def _mask_data(
        data: dict[str, Tensor] | None,
        mask: Tensor | None,
    ) -> dict[str, Tensor] | None:
        """Filter a data dict by `mask` along the leading axis.

        Tensors whose leading dim matches mask are sliced; others pass through.
        None data or mask is a no-op.
        """
        if data is None or mask is None:
            return data
        n = mask.shape[0]
        out: dict[str, Tensor] = {}
        for k, v in data.items():
            if v.ndim > 0 and v.shape[0] == n:
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    def _record_round(self, t: int) -> None:
        theta = self.predictor.model.get_params().detach().cpu()
        record: dict[str, Any] = {"round": t, "theta": theta}
        if self._prev_theta is not None:
            record["stability_gap"] = stability_gap(self._prev_theta, theta)
        with torch.no_grad():
            for name, fn in self.metrics.items():
                record[name] = fn(self)
        if self.dataset is not None:
            record["dataset_hash"] = self.dataset.hash()
        self.history.append(**record)
        self._prev_theta = theta
