"""Tests for the AgentTorch adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("agent_torch")

import torch
from torch import Tensor

from perfsim.adapters.agenttorch import (
    AgentTorchEnvironment,
    SignalMutationError,
)
from perfsim.core.environment import AgentBased
from perfsim.core.model import Model
from perfsim.learners.erm import ERMLearner
from perfsim.losses import MSELoss
from perfsim.models.linear import LinearModel
from perfsim.simulator import Simulator


# ---- Stubs ----------------------------------------------------------------


class IdentityModel(Model):
    """Returns the input feature column unchanged (squeezed). Used to make
    predictions deterministic in tests."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(-1) if x.ndim == 2 and x.shape[-1] == 1 else x


class FakeRunner:
    """Mimics the subset of agent_torch.Runner that the adapter touches.

    Two transition modes:
        mode="a2":  inner loop reads platform_signal, evolves opinion via
                    opinion = peer_sus * platform_signal + (1 - peer_sus) * opinion
                    leaving platform_signal untouched. Pattern A2.
        mode="b":   inner loop overwrites platform_signal (geometric decay).
                    Pattern B. Triggers SignalMutationError under strict_signal=True.
        mode="noop": runner.step is a no-op; opinion and signal unchanged.
    """

    def __init__(self, n: int = 4, mode: str = "a2", seed: int = 0) -> None:
        self._mode = mode
        self._seed = seed
        self._n = n
        torch.manual_seed(seed)
        features = torch.linspace(0.1, 0.9, n).reshape(-1, 1)
        opinion = features.squeeze(-1).clone()
        signal = torch.zeros(n)
        peer_sus = torch.full((n,), 0.5)
        self.state = {
            "agents": {
                "citizen": {
                    "features": features,
                    "opinion": opinion,
                    "platform_signal": signal,
                    "peer_sus": peer_sus,
                }
            },
            "current_step": 0,
        }
        # Track how many substep advances happened across the runner's life,
        # for the keep_trajectory test.
        self._substep_log: list[int] = []

    def reset_state_before_episode(self) -> None:
        self._substep_log = []

    def step(self, num_steps: int | None = None) -> None:
        if num_steps is None:
            num_steps = 1
        for _ in range(num_steps):
            agent = self.state["agents"]["citizen"]
            x = agent["opinion"]
            sig = agent["platform_signal"]
            peer = agent["peer_sus"]
            if self._mode == "a2":
                # Fixed-anchor FJ-style update; signal untouched.
                agent["opinion"] = peer * sig + (1 - peer) * x
            elif self._mode == "b":
                # B violation: substep overwrites the signal.
                agent["platform_signal"] = 0.9 * sig
                agent["opinion"] = peer * sig + (1 - peer) * x
            elif self._mode == "noop":
                pass
            else:
                raise ValueError(f"unknown mode {self._mode!r}")
            self.state["current_step"] = self.state["current_step"] + 1
            self._substep_log.append(self.state["current_step"])


# ---- Helpers --------------------------------------------------------------


def _writer(runner: FakeRunner, preds: Tensor) -> None:
    # User-supplied writers are responsible for matching the AT sim's
    # expected signal shape. Our FakeRunner keeps platform_signal as (N,).
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["citizen"]["platform_signal"] = preds.clone()


def _feature(runner: FakeRunner) -> Tensor:
    return runner.state["agents"]["citizen"]["features"]


def _extract(runner: FakeRunner):
    agent = runner.state["agents"]["citizen"]
    n = agent["opinion"].shape[0]
    return {
        "x": agent["features"],
        "y": agent["opinion"].unsqueeze(-1),
        "agent_idx": torch.arange(n),
    }


SIGNAL_PATH = ("agents", "citizen", "platform_signal")


def _make_env(mode: str = "a2", **kwargs) -> AgentTorchEnvironment:
    return AgentTorchEnvironment(
        runner_factory=lambda seed: FakeRunner(mode=mode, seed=seed),
        feature_provider=_feature,
        signal_writer=_writer,
        state_extractor=_extract,
        signal_path=SIGNAL_PATH,
        **kwargs,
    )


# ---- Type and construction ------------------------------------------------


def test_inherits_agent_based():
    env = _make_env()
    assert isinstance(env, AgentBased)


def test_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="runner_factory"):
        AgentTorchEnvironment(
            runner_factory="not a callable",  # type: ignore[arg-type]
            feature_provider=_feature,
            signal_writer=_writer,
            state_extractor=_extract,
            signal_path=SIGNAL_PATH,
        )


def test_rejects_non_callable_signal_writer():
    with pytest.raises(TypeError, match="signal_writer"):
        AgentTorchEnvironment(
            runner_factory=lambda seed: FakeRunner(),
            feature_provider=_feature,
            signal_writer="nope",  # type: ignore[arg-type]
            state_extractor=_extract,
            signal_path=SIGNAL_PATH,
        )


def test_rejects_empty_signal_path():
    with pytest.raises(TypeError, match="signal_path"):
        AgentTorchEnvironment(
            runner_factory=lambda seed: FakeRunner(),
            feature_provider=_feature,
            signal_writer=_writer,
            state_extractor=_extract,
            signal_path=(),
        )


# ---- run / step / sample --------------------------------------------------


def test_run_a2_returns_supervised_data():
    env = _make_env(mode="a2")
    data = env.run(IdentityModel(), n_steps=3)
    assert set(data.keys()) >= {"x", "y", "agent_idx"}
    assert data["y"].shape == (4, 1)
    assert data["agent_idx"].tolist() == [0, 1, 2, 3]


def test_run_writes_signal_and_advances():
    env = _make_env(mode="a2")
    runner = env.runner
    before = runner.state["agents"]["citizen"]["platform_signal"].clone()
    env.run(IdentityModel(), n_steps=2)
    after = runner.state["agents"]["citizen"]["platform_signal"]
    # signal_writer ran (predictions are nonzero), so post-run signal differs
    # from initial zero state.
    assert not torch.allclose(before, after)


def test_step_delegates_to_run_n1():
    env = _make_env(mode="a2")
    data = env.step(IdentityModel())
    assert data["y"].shape == (4, 1)


def test_run_rejects_invalid_n_steps():
    env = _make_env(mode="a2")
    with pytest.raises(ValueError, match="positive int"):
        env.run(IdentityModel(), n_steps=0)
    with pytest.raises(ValueError, match="positive int"):
        env.run(IdentityModel(), n_steps=-3)


def test_sample_raises_not_implemented():
    env = _make_env(mode="a2")
    with pytest.raises(NotImplementedError, match="not supported"):
        env.sample(IdentityModel())


def test_grad_sample_raises_not_implemented():
    env = _make_env(mode="a2")
    with pytest.raises(NotImplementedError):
        env.grad_sample(IdentityModel())


# ---- B-violation detection ------------------------------------------------


def test_b_violation_raises_under_strict_signal():
    env = _make_env(mode="b", strict_signal=True)
    with pytest.raises(SignalMutationError, match="read-only"):
        env.run(IdentityModel(), n_steps=5)


def test_b_violation_silenced_when_strict_signal_false():
    env = _make_env(mode="b", strict_signal=False)
    # Should run without raising.
    data = env.run(IdentityModel(), n_steps=5)
    assert "y" in data


# ---- reset / seeding ------------------------------------------------------


def test_reset_replaces_runner():
    env = _make_env(mode="a2", init_seed=0)
    first = env.runner
    env.reset(seed=42)
    second = env.runner
    assert first is not second


def test_run_advances_internal_step_counter():
    env = _make_env(mode="a2")
    runner = env.runner
    env.run(IdentityModel(), n_steps=5)
    assert runner.state["current_step"] == 5


def test_run_truncates_trajectory_log_by_default():
    env = _make_env(mode="a2")
    runner = env.runner
    env.run(IdentityModel(), n_steps=3)
    # First run: log length 3.
    assert len(runner._substep_log) == 3
    env.run(IdentityModel(), n_steps=2)
    # Second run: truncated, then 2 steps logged. Not 5.
    assert len(runner._substep_log) == 2


def test_run_preserves_trajectory_log_when_keep_trajectory():
    env = _make_env(mode="a2", keep_trajectory=True)
    runner = env.runner
    env.run(IdentityModel(), n_steps=3)
    env.run(IdentityModel(), n_steps=2)
    # No truncation between epochs.
    assert len(runner._substep_log) == 5


# ---- max_meaningful_epoch_size ---------------------------------------------


def test_per_instance_epoch_size_default_inf():
    env = _make_env(mode="a2")
    assert env.max_meaningful_epoch_size == float("inf")


def test_per_instance_epoch_size_override():
    env = _make_env(mode="a2", max_meaningful_epoch_size=1)
    assert env.max_meaningful_epoch_size == 1


# ---- produces_schema -------------------------------------------------------


def test_produces_supervised_schema_by_default():
    env = _make_env(mode="a2")
    assert env.produces_schema.name == "supervised"


# ---- end-to-end smoke against Simulator -----------------------------------


def test_simulator_drives_at_env_through_one_epoch():
    """End-to-end: real Simulator + adapter + FakeRunner. Confirms the
    adapter's run() contract integrates with the Simulator hot path.
    """
    env = _make_env(mode="a2")
    model = LinearModel(in_features=1, out_features=1)
    loss = MSELoss()
    learner = ERMLearner(model=model, loss=loss)

    sim = Simulator(env=env, learner=learner, loss=loss)
    hist = sim.run(n_rounds=2, epoch_size=3, seed=0)
    assert len(hist) == 2
    # theta logged at each round.
    assert "theta" in hist[0]
