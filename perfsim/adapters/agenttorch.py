"""AgentTorch adapter: wraps an `agent_torch.Runner` as a perfsim `AgentBased` env.

Algorithm 1 (arxiv 2603.12137): the deployed model is queried once at the top
of `env.run`, its predictions are written into AT state via `signal_writer`,
then the runner advances n_steps internally without re-querying. The signal
field must stay read-only across the inner loop; `strict_signal` asserts this.
The adapter satisfies `Differentiable` (grad through the single query) but not
`FullyDifferentiable` (theta is frozen across the epoch loop).
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import torch
from torch import Tensor

import agent_torch  # noqa: F401
from agent_torch.core.runner import Runner as ATRunner

from perfsim.core.environment import AgentBased
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema


RunnerFactory = Callable[[int], ATRunner]
FeatureProvider = Callable[[ATRunner], Tensor]
SignalWriter = Callable[[ATRunner, Tensor], None]
StateExtractor = Callable[[ATRunner], Data]
SignalPath = tuple[str, ...]


class SignalMutationError(RuntimeError):
    """Raised when the AT sim mutated the signal field during the inner loop."""


class AgentTorchEnvironment(AgentBased):
    """Wraps an agent_torch.Runner via runner_factory + 3 callables + signal_path.

    The user supplies feature_provider (state -> X), signal_writer (state, preds
    -> mutate state), state_extractor (state -> Data), and signal_path (the
    state-dict key path to the signal field that must stay read-only).
    """

    max_meaningful_epoch_size: ClassVar[int | float] = float("inf")

    def __init__(
        self,
        runner_factory: RunnerFactory,
        *,
        feature_provider: FeatureProvider,
        signal_writer: SignalWriter,
        state_extractor: StateExtractor,
        signal_path: SignalPath,
        produces_schema: DataSchema = SUPERVISED_SCHEMA,
        max_meaningful_epoch_size: int | float = float("inf"),
        keep_trajectory: bool = False,
        strict_signal: bool = True,
        init_seed: int = 0,
    ) -> None:
        if not callable(runner_factory):
            raise TypeError("runner_factory must be callable")
        if not callable(feature_provider):
            raise TypeError("feature_provider must be callable")
        if not callable(signal_writer):
            raise TypeError("signal_writer must be callable")
        if not callable(state_extractor):
            raise TypeError("state_extractor must be callable")
        if not isinstance(signal_path, tuple) or not signal_path:
            raise TypeError(
                "signal_path must be a non-empty tuple of state-dict keys; "
                f"got {signal_path!r}"
            )

        self._factory = runner_factory
        self._feature_provider = feature_provider
        self._signal_writer = signal_writer
        self._state_extractor = state_extractor
        self._signal_path = signal_path
        self._produces_schema = produces_schema
        # Per-instance so the Simulator's epoch_size validator reads the
        # constructor value, not the class default.
        self.max_meaningful_epoch_size = max_meaningful_epoch_size
        self._keep_trajectory = bool(keep_trajectory)
        self._strict_signal = bool(strict_signal)

        self._runner: ATRunner = self._build_runner(init_seed)

    @property
    def produces_schema(self) -> DataSchema:
        return self._produces_schema

    @property
    def runner(self) -> ATRunner:
        """The held agent_torch.Runner, exposed for debug / inspection."""
        return self._runner

    @property
    def signal_path(self) -> SignalPath:
        return self._signal_path

    def _build_runner(self, seed: int) -> ATRunner:
        # Duck-typed on .state / .step / .reset_state_before_episode so tests
        # can pass stubs. The factory is expected to have called runner.init().
        return self._factory(int(seed))

    def reset(self, seed: int = 0) -> None:
        """Rebuild a fresh runner via the factory (the only supported re-seed)."""
        self._runner = self._build_runner(seed)

    def sample(self, model: Model) -> Data:
        """Not supported; the hot path uses `run`, not `sample`."""
        raise NotImplementedError(
            "AgentTorchEnvironment.sample is not supported. "
            "Use env.run(model, n_steps) instead."
        )

    def step(self, model: Model) -> Data:
        """One AT time step; wrapper around run(model, n_steps=1)."""
        return self.run(model, n_steps=1)

    def run(self, model: Model, n_steps: int) -> Data:
        """Query the model once, write the signal, advance n_steps, extract Data."""
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")

        runner = self._runner
        if not self._keep_trajectory and runner.state is not None:
            runner.reset_state_before_episode()

        X = self._feature_provider(runner)
        with torch.no_grad():
            preds = model(X)
        preds = preds.detach()
        self._signal_writer(runner, preds)

        sig_before: Tensor | None = None
        if self._strict_signal:
            sig_before = self._read_signal().clone()

        runner.step(num_steps=n_steps)

        if self._strict_signal and sig_before is not None:
            sig_after = self._read_signal()
            if not torch.allclose(sig_before, sig_after):
                raise SignalMutationError(
                    f"Signal field {self._signal_path!r} was mutated during "
                    f"runner.step(num_steps={n_steps}). AT substeps must treat "
                    f"the signal field as read-only across the inner loop. "
                    f"Pass strict_signal=False to disable this check."
                )

        return self._state_extractor(runner)

    def grad_sample(self, model: Model) -> Data:
        raise NotImplementedError(
            "AgentTorchEnvironment.grad_sample is not supported."
        )

    def grad_step(self, model: Model) -> Data:
        """Like step but with gradients live; wrapper around grad_run(., 1)."""
        return self.grad_run(model, n_steps=1)

    def grad_run(self, model: Model, n_steps: int) -> Data:
        """Like run but with gradients live: model(X) runs without no_grad and
        preds are not detached. Grad through runner.step depends on the AT sim's
        differentiability (covid: yes, via StraightThroughBernoulli).
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")

        runner = self._runner
        if not self._keep_trajectory and runner.state is not None:
            runner.reset_state_before_episode()

        X = self._feature_provider(runner)
        preds = model(X)
        self._signal_writer(runner, preds)

        sig_before: Tensor | None = None
        if self._strict_signal:
            sig_before = self._read_signal().detach().clone()

        runner.step(num_steps=n_steps)

        if self._strict_signal and sig_before is not None:
            sig_after = self._read_signal().detach()
            if not torch.allclose(sig_before, sig_after):
                raise SignalMutationError(
                    f"Signal field {self._signal_path!r} was mutated during "
                    f"runner.step(num_steps={n_steps}) in grad_run."
                )

        return self._state_extractor(runner)

    def _read_signal(self) -> Tensor:
        """Walk runner.state along signal_path and return the leaf tensor."""
        cursor: Any = self._runner.state
        if cursor is None:
            raise RuntimeError(
                "runner.state is None. Did the runner_factory forget to call "
                "runner.init() before returning?"
            )
        for key in self._signal_path:
            cursor = cursor[key]
        if not isinstance(cursor, Tensor):
            raise TypeError(
                f"signal_path {self._signal_path!r} resolved to "
                f"{type(cursor).__name__}, expected torch.Tensor"
            )
        return cursor
