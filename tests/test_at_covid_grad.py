"""Gradient-mode tests for the at_covid scenario.

Asserts that `env.grad_run(model, n_steps=K)` produces a loss whose backward
yields non-zero gradient on the predictor's parameters.

Slow: each test builds an AT runner (~5s init) and runs K inner substeps.
Marked `slow` and excluded by default; run with `-m slow`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agent_torch")

from perfsim.scenarios.at_covid._compat import bundled_astoria_dir, bundled_covid_yaml  # noqa: E402

if not (bundled_astoria_dir() / "age.pickle").exists():
    pytest.skip("Bundled astoria population missing", allow_module_level=True)
if not bundled_covid_yaml().exists():
    pytest.skip("Bundled covid YAML missing", allow_module_level=True)

import torch  # noqa: E402

from perfsim.scenarios.at_covid import (  # noqa: E402
    default_signal_writer_grad,
    make_covid_env,
    seed_initial_infections,
)


# ---- seed_initial_infections shape contract -------------------------------


def test_seed_initial_infections_marks_correct_count():
    env = make_covid_env(init_seed=0)
    n_total = env.runner.state["agents"]["citizens"]["disease_stage"].shape[0]
    n_seeded = seed_initial_infections(env, fraction=0.10, seed=0)
    assert n_seeded > 0
    assert n_seeded < n_total
    ds = env.runner.state["agents"]["citizens"]["disease_stage"]
    assert int((ds == 2.0).sum().item()) == n_seeded


def test_seed_initial_infections_accepts_runner_directly():
    env = make_covid_env(init_seed=0)
    n_seeded = seed_initial_infections(env.runner, fraction=0.05, seed=0)
    assert n_seeded > 0


def test_seed_initial_infections_rejects_invalid_fraction():
    env = make_covid_env(init_seed=0)
    with pytest.raises(ValueError, match="fraction"):
        seed_initial_infections(env, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        seed_initial_infections(env, fraction=1.5)


def test_seed_initial_infections_resets_infected_time():
    env = make_covid_env(init_seed=0)
    seed_initial_infections(env, fraction=0.10, seed=0)
    citizens = env.runner.state["agents"]["citizens"]
    infected = citizens["disease_stage"].squeeze() == 2.0
    assert torch.all(citizens["infected_time"].squeeze()[infected] == 0)


# ---- gradient flow --------------------------------------------------------


@pytest.mark.slow
def test_grad_run_yields_nonzero_gradient():
    env = make_covid_env(
        init_seed=0,
        signal_writer=default_signal_writer_grad,
    )
    seed_initial_infections(env, fraction=0.05, seed=0)

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.05)
        model.bias.fill_(-1.0)

    env.grad_run(model, n_steps=5)
    loss = env.runner.state["environment"]["daily_infected"].sum()
    assert loss.requires_grad
    loss.backward()

    assert model.weight.grad is not None
    assert model.bias.grad is not None
    assert model.weight.grad.norm().item() > 0.0
    assert model.bias.grad.norm().item() > 0.0


@pytest.mark.slow
def test_run_does_not_propagate_grad_to_predictor_params():
    """`run` wraps `model(X)` in `torch.no_grad`, so even though AT
    substeps have their own learnable params (loss.requires_grad will be
    True via the AT side), the perfsim predictor's params receive no
    gradient from a backward on AT state.
    """
    env = make_covid_env(init_seed=0)
    seed_initial_infections(env, fraction=0.05, seed=0)

    model = torch.nn.Linear(1, 1)

    env.run(model, n_steps=2)
    loss = env.runner.state["environment"]["daily_infected"].sum()
    # AT-side learnables may produce a graph; that is fine. What we
    # require is that the predictor's params are NOT in that graph.
    if loss.requires_grad:
        try:
            loss.backward()
        except RuntimeError:
            # AT learnables may not retain their own graph either; we
            # only care that model.weight.grad stays None.
            pass
    assert model.weight.grad is None or model.weight.grad.norm().item() == 0.0
    assert model.bias.grad is None or model.bias.grad.norm().item() == 0.0
