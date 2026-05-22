"""Tests for the perfsim.scenarios.at_covid package.

Gated by `pytest.importorskip("agent_torch")` and by the bundled astoria
population being present. Two tiers:

- Fast tests check shape/typing of the scenario API without touching the
  real Runner (no init, no step). Run on every invocation.
- One slow test (`test_smoke_one_round`) actually builds a real
  agent_torch.Runner and runs a single perfsim epoch. Marked `slow`.
  ~13s wall clock. Excluded by default; run with `-m slow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("agent_torch")

# Skip the entire module if the bundled astoria population is missing
# (which it would be in a fresh container that did not install the full
# agent_torch wheel with data).
from perfsim.scenarios.at_covid._compat import bundled_astoria_dir, bundled_covid_yaml  # noqa: E402

if not (bundled_astoria_dir() / "age.pickle").exists():
    pytest.skip(
        "Bundled astoria population missing; install agent-torch with data",
        allow_module_level=True,
    )
if not bundled_covid_yaml().exists():
    pytest.skip(
        "Bundled covid YAML missing", allow_module_level=True
    )

import torch  # noqa: E402

from perfsim.adapters.agenttorch import AgentTorchEnvironment  # noqa: E402
from perfsim.scenarios.at_covid import (  # noqa: E402
    PerfsimIsolationDecision,
    build_covid_runner,
    default_feature_provider,
    default_signal_writer,
    default_state_extractor,
    make_covid_env,
)


# ---- Package surface ------------------------------------------------------


def test_public_exports_are_callable_or_class():
    assert callable(make_covid_env)
    assert callable(build_covid_runner)
    assert callable(default_feature_provider)
    assert callable(default_signal_writer)
    assert callable(default_state_extractor)
    assert isinstance(PerfsimIsolationDecision, type)


def test_make_covid_env_returns_adapter():
    env = make_covid_env(init_seed=0)
    assert isinstance(env, AgentTorchEnvironment)


def test_make_covid_env_signal_path():
    env = make_covid_env(init_seed=0)
    assert env.signal_path == ("agents", "citizens", "platform_signal")


def test_make_covid_env_produces_supervised_schema():
    env = make_covid_env(init_seed=0)
    assert env.produces_schema.name == "supervised"


def test_make_covid_env_strict_signal_default():
    env = make_covid_env(init_seed=0)
    assert env._strict_signal is True


# ---- Runner factory shape contract ---------------------------------------


def test_build_covid_runner_state_has_platform_signal():
    runner = build_covid_runner(seed=0)
    citizens = runner.state["agents"]["citizens"]
    assert "platform_signal" in citizens
    n = citizens["age"].shape[0]
    assert citizens["platform_signal"].shape == (n,)
    # Seeded as zeros.
    assert torch.allclose(
        citizens["platform_signal"], torch.zeros_like(citizens["platform_signal"])
    )


def test_build_covid_runner_state_has_required_fields():
    runner = build_covid_runner(seed=0)
    citizens = runner.state["agents"]["citizens"]
    for key in ("age", "disease_stage", "id", "is_quarantined"):
        assert key in citizens, f"missing citizens.{key}"


# ---- Default callables ---------------------------------------------------


def test_default_feature_provider_returns_age():
    runner = build_covid_runner(seed=0)
    x = default_feature_provider(runner)
    assert x.dtype == torch.float32
    assert x.shape == runner.state["agents"]["citizens"]["age"].shape


def test_default_signal_writer_squeezes_2d_input():
    runner = build_covid_runner(seed=0)
    n = runner.state["agents"]["citizens"]["age"].shape[0]
    preds = torch.linspace(-1.0, 1.0, n).reshape(-1, 1)
    default_signal_writer(runner, preds)
    stored = runner.state["agents"]["citizens"]["platform_signal"]
    assert stored.ndim == 1
    assert stored.shape == (n,)
    assert torch.allclose(stored, preds.squeeze(-1))


def test_default_state_extractor_returns_supervised_data():
    runner = build_covid_runner(seed=0)
    data = default_state_extractor(runner)
    assert set(data.keys()) >= {"x", "y", "agent_idx"}
    n = data["x"].shape[0]
    assert data["y"].shape == (n, 1)
    assert data["agent_idx"].tolist() == list(range(n))


# ---- Reset reseeds the runner --------------------------------------------


def test_reset_replaces_runner():
    env = make_covid_env(init_seed=0)
    first = env.runner
    env.reset(seed=42)
    second = env.runner
    assert first is not second


# ---- Slow end-to-end test -------------------------------------------------


@pytest.mark.slow
def test_smoke_one_round():
    """Real Runner, real Simulator, one round of one epoch step. ~10-15s."""
    from perfsim.learners.erm import ERMLearner
    from perfsim.losses import MSELoss
    from perfsim.models.linear import LinearModel
    from perfsim.simulator import Simulator

    env = make_covid_env(init_seed=0)
    model = LinearModel(in_features=1, out_features=1)
    loss = MSELoss()
    learner = ERMLearner(model=model, loss=loss, max_iter=5)
    sim = Simulator(env=env, learner=learner, loss=loss)

    hist = sim.run(n_rounds=1, epoch_size=1, seed=0)
    assert len(hist) == 1
    assert "theta" in hist[0]
    # Signal field stayed read-only across the inner loop (A2 / strict_signal pass).
    # If the signal was mutated we would have hit SignalMutationError, not gotten here.
