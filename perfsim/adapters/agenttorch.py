"""AgentTorch adapter: wraps an `agent_torch.Runner` as a perfsim `AgentBased`
environment.

See DESIGN.md §20 for the full design rationale. Short version:

- Algorithm 1 (arxiv 2603.12137): the deployed perfsim model is queried ONCE
  at the top of `env.run`. Its predictions are written into AT state via a
  user-supplied `signal_writer`. The AT runner then advances `n_steps` time
  steps internally without re-querying the model.

- The adapter is pure plumbing. The user supplies three callables
  (`feature_provider`, `signal_writer`, `state_extractor`) plus a
  `runner_factory(seed)` for reset/seeding. perfsim knows nothing about
  AT-side config or substep code.

- Pattern contract:
    A1 (one-shot seed): predictions used at step 0, then dropped. OK.
    A2 (fixed anchor): predictions read every substep, never overwritten. OK.
    B  (signal mutated): some substep overwrites the signal field. Forbidden.
  The adapter snapshots the signal field before `runner.step(...)` and
  asserts `torch.allclose` after, flagging B violations. Disable with
  `strict_signal=False`.

- The adapter satisfies `Differentiable` only (gradients can flow through
  the single per-epoch model query). It does NOT satisfy `FullyDifferentiable`:
  the Simulator's epoch loop freezes theta across the inner loop (DESIGN.md
  §8), so gradients wrt theta over a full rollout are not supported in v1.

- Install: `pip install 'perfsim[agenttorch]'`. The module imports `agent_torch`
  at load time; if the extra is missing, you get an ImportError pointing to
  the install hint.
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
    """Raised when the AT sim mutated the platform-signal field during an
    epoch's inner loop (pattern B violation; DESIGN.md §20).
    """


class AgentTorchEnvironment(AgentBased):
    """Wraps an agent_torch.Runner as a perfsim AgentBased environment.

    Construction:

        env = AgentTorchEnvironment(
            runner_factory  = lambda seed: build_my_at_runner(seed),
            feature_provider= lambda r: r.state["agents"]["citizen"]["features"],
            signal_writer   = lambda r, p: r.state["agents"]["citizen"].__setitem__("platform_signal", p),
            state_extractor = lambda r: {
                "x": r.state["agents"]["citizen"]["features"],
                "y": r.state["agents"]["citizen"]["opinion"],
                "agent_idx": torch.arange(r.state["agents"]["citizen"]["opinion"].shape[0]),
            },
            signal_path = ("agents", "citizen", "platform_signal"),
        )
   
    env.run(model, n_steps):
        if not keep_trajectory: runner.reset_state_before_episode()
        X = feature_provider(runner)
        preds = model(X) # queried ONCE
        sig_before = read(runner.state, signal_path)
        signal_writer(runner, preds)  write into runner.state
        runner.step(num_steps=n_steps) # AT advances internally
        assert torch.allclose(sig_before_overwritten_by_preds, sig_after)
        return state_extractor(runner)
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
        # Stored per-instance so the Simulator's epoch_size validator (which
        # reads getattr(env, 'max_meaningful_epoch_size', ...)) sees the
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
        """The held `agent_torch.Runner`. Exposed for debug / inspection."""
        return self._runner

    @property
    def signal_path(self) -> SignalPath:
        return self._signal_path

    def _build_runner(self, seed: int) -> ATRunner:
        runner = self._factory(int(seed))
        # AT's Runner constructor does NOT call init(); the user typically
        # calls runner.init() before use. We do not call it here because the
        # factory may have done so already. If state is None, run() will
        # fail loudly at the first state read, which is the right behavior.
        # No isinstance check: duck-typing on .state, .step(num_steps=...),
        # .reset_state_before_episode() is enough, and lets tests use stubs.
        return runner

    def reset(self, seed: int = 0) -> None:
        """Discard the current runner and construct a fresh one via the
        factory. This is the only supported way to re-seed; AT's own
        `Runner.reset()` does not re-seed any RNG.
        """
        self._runner = self._build_runner(seed)

    def sample(self, model: Model) -> Data:
        """Not supported in v1. AT runners do not expose a free peek
        primitive; the Simulator's hot path uses `run`, not `sample`.
        """
        raise NotImplementedError(
            "AgentTorchEnvironment.sample is not supported in v1 "
            "(DESIGN.md §20). Use env.run(model, n_steps) instead."
        )

    def step(self, model: Model) -> Data:
        """One AT time step (= one pass through all substeps).

        Convenience wrapper around `self.run(model, n_steps=1)`. AT's step
        is the elementary unit; there is no smaller granularity to expose.
        """
        return self.run(model, n_steps=1)

    def run(self, model: Model, n_steps: int) -> Data:
        """Algorithm 1 epoch: query the model once, write the signal,
        advance AT for `n_steps` time steps, return the final-state Data.

        Raises:
            SignalMutationError: if `strict_signal=True` and the signal
              field changed during the inner loop (pattern B violation).
            ValueError: if `n_steps < 1`.
        """
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
                    f"runner.step(num_steps={n_steps}). This violates the "
                    f"Algorithm 1 contract: AT substeps must treat the signal "
                    f"field as read-only across the inner loop (DESIGN.md §20 "
                    f"pattern B). Pass strict_signal=False to disable this "
                    f"check."
                )

        return self._state_extractor(runner)


    def grad_sample(self, model: Model) -> Data:
        raise NotImplementedError(
            "AgentTorchEnvironment.grad_sample is not supported in v1 "
            "(see DESIGN.md §20: sample() is unsupported in v1; the "
            "differentiable variant inherits that limitation)."
        )

    def grad_step(self, model: Model) -> Data:
        """Same shape as `step` but with gradients live (no `torch.no_grad`).

        Convenience wrapper around `self.grad_run(model, n_steps=1)`.
        """
        return self.grad_run(model, n_steps=1)

    def grad_run(self, model: Model, n_steps: int) -> Data:
        """Same shape as `run` but with gradients live.

        Differences from `run`:
          - `model(X)` is invoked WITHOUT `torch.no_grad`; the autograd
            graph from model params to preds is preserved.
          - Predictions are not explicitly detached before being passed to
            `signal_writer`. The user-supplied `signal_writer` must not
            detach either; the at_covid defaults already use `.clone()`
            without `.detach()` so they are grad-safe.

        Gradient through `runner.step(num_steps=n_steps)` depends on the
        AT sim's differentiability. For covid: yes through
        `StraightThroughBernoulli` + `update_stages`; see the at_covid
        README for the four conditions needed for non-zero gradient.

        The strict-signal check still fires if the AT sim mutates the
        signal field (pattern B). Snapshots are detached before the
        allclose, so the check itself does not consume gradient.
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
                    f"runner.step(num_steps={n_steps}) in grad_run "
                    f"(DESIGN.md §20 pattern B)."
                )

        return self._state_extractor(runner)

    # ---- Internal helpers --------------------------------------------------

    def _read_signal(self) -> Tensor:
        """Walk `runner.state` along `signal_path` and return the leaf tensor."""
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
