"""Build an `AgentTorchEnvironment` wired to AT's bundled covid model.

Public entry point: `make_covid_env(seed=0, **overrides) -> AgentTorchEnvironment`.

Wires:
  - `runner_factory` that builds an `agent_torch.Runner` with the bundled
    covid YAML (population_dir patched to the bundled astoria path), our
    `PerfsimIsolationDecision` action substep registered in place of the
    bundled one, and `platform_signal` initialized to zeros on
    `agents/citizens`.
  - Default `feature_provider`, `signal_writer`, `state_extractor` that use
    `age` as the feature and `disease_stage` as the supervised target. Any
    of the three can be overridden via kwargs.

The example at `examples/at_covid_smoke.py` is a thin caller of this module.
"""

from __future__ import annotations

import tempfile
from typing import Callable, Optional

import torch
from torch import Tensor

from perfsim.scenarios.at_covid._compat import (
    bundled_astoria_dir,
    bundled_covid_yaml,
    install_langchain_shim,
    should_register_resolvers,
)


# We install the langchain shim before importing AT covid substep modules.
install_langchain_shim()

from agent_torch.core import Registry, Runner  # noqa: E402
from agent_torch.core.helpers import read_config  # noqa: E402
from agent_torch.models.covid.substeps.new_transmission.transition import (  # noqa: E402
    NewTransmission,
)
from agent_torch.models.covid.substeps.seirm_progression.transition import (  # noqa: E402
    SEIRMProgression,
)
from agent_torch.models.covid.substeps.utils import (  # noqa: E402
    get_infected_time,
    get_lam_gamma_integrals,
    get_mean_agent_interactions,
    get_next_stage_time,
    initialize_id,
    load_population_attribute,
    network_from_file,
    read_from_file,
)

from perfsim.adapters.agenttorch import AgentTorchEnvironment  # noqa: E402
from perfsim.scenarios.at_covid.action import PerfsimIsolationDecision  # noqa: E402


# ---- Default callables ----------------------------------------------------


def default_feature_provider(runner: Runner) -> Tensor:
    """Per-agent age as a (N, 1) float feature."""
    return runner.state["agents"]["citizens"]["age"].float().detach()


def default_signal_writer(runner: Runner, preds: Tensor) -> None:
    """Squeeze predictions to (N,) and store at `agents/citizens/platform_signal`.

    Detaches before writing so that the non-grad `run` path is the safe
    default. For gradient measurement through `grad_run`, pass
    `default_signal_writer_grad` instead via `make_covid_env(signal_writer=...)`.
    """
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["citizens"]["platform_signal"] = preds.detach().clone()


def default_signal_writer_grad(runner: Runner, preds: Tensor) -> None:
    """Non-detaching variant for use with `grad_run`.

    Preserves the autograd graph from the predictor's params through the
    signal field. The Simulator's regular `run` path is unaffected because
    the adapter wraps `model(X)` in `torch.no_grad`, so preds is already
    grad-free in that case and `.clone()` is a no-op for the graph.
    """
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["citizens"]["platform_signal"] = preds.clone()


def default_state_extractor(runner: Runner) -> dict[str, Tensor]:
    """Return `(x=age, y=disease_stage, agent_idx)` as perfsim supervised Data."""
    citizens = runner.state["agents"]["citizens"]
    age = citizens["age"].float().detach()
    disease_stage = citizens["disease_stage"].float().detach().reshape(-1, 1)
    n = age.shape[0]
    return {
        "x": age,
        "y": disease_stage,
        "agent_idx": torch.arange(n),
    }



def seed_initial_infections(
    runner_or_env, fraction: float = 0.05, seed: int = 0
) -> int:
    """Mark `fraction` of agents as infected (disease_stage = INFECTED) and reset
    their `infected_time` to 0.

    Without this, the population starts all-susceptible and `runner.step(1)`
    produces zero new exposures (every Bernoulli draws on probability ~0).
    The autograd graph is intact but the gradient is numerically zero. Seeding
    a small fraction of the population gives the dynamics something to chew on
    and unlocks usable gradient signal through `grad_run`.

    Accepts either a `Runner` directly or an `AgentTorchEnvironment` (in which
    case it seeds the env's held runner). Returns the number of agents
    actually marked infected.

    INFECTED is disease_stage = 2 in covid's SEIRM encoding (S=0, E=1, I=2,
    R=3, M=4).
    """
    runner = runner_or_env.runner if isinstance(runner_or_env, AgentTorchEnvironment) else runner_or_env

    citizens = runner.state["agents"]["citizens"]
    n = citizens["disease_stage"].shape[0]

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction!r}")

    gen = torch.Generator().manual_seed(int(seed))
    mask = torch.rand(n, 1, generator=gen) < float(fraction)

    INFECTED = 2.0
    ds_new = citizens["disease_stage"].clone()
    ds_new[mask] = INFECTED
    citizens["disease_stage"] = ds_new

    if "infected_time" in citizens:
        it_new = citizens["infected_time"].clone()
        it_new[mask] = 0
        citizens["infected_time"] = it_new

    return int(mask.sum().item())


def _patched_covid_yaml() -> str:
    """Rewrite the bundled covid YAML so `population_dir` resolves to the
    bundled astoria path (the YAML otherwise hardcodes the AT authors'
    machine path). Returns the path to a temp file holding the patched YAML.
    """
    src = bundled_covid_yaml()
    text = src.read_text()
    text = text.replace(
        "/u/ayushc/projects/GradABM/systems/AgentTorch/agent_torch/populations/astoria",
        str(bundled_astoria_dir()),
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def _build_registry() -> Registry:
    reg = Registry()
    reg.register(PerfsimIsolationDecision, "make_isolation_decision", key="policy")
    reg.register(NewTransmission, "new_transmission", key="transition")
    reg.register(SEIRMProgression, "seirm_progression", key="transition")
    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(read_from_file, "read_from_file", key="initialization")
    reg.register(get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(
        get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization"
    )
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")
    reg.register(load_population_attribute, "load_population_attribute", key="initialization")
    reg.register(initialize_id, "initialize_id", key="initialization")
    return reg


def build_covid_runner(seed: int = 0) -> Runner:
    """Construct a fully-initialized `agent_torch.Runner` for the bundled
    covid model.

    Seeds `torch.manual_seed` before init for reproducibility. Adds a
    zero-initialized `platform_signal` tensor on `agents/citizens` so the
    perfsim adapter's first `signal_writer` call has a slot to overwrite.
    """
    torch.manual_seed(int(seed))
    yaml_path = _patched_covid_yaml()
    config = read_config(yaml_path, register_resolvers=should_register_resolvers())
    reg = _build_registry()
    runner = Runner(config, reg)
    runner.init()

    n = config["simulation_metadata"]["num_agents"]
    runner.state["agents"]["citizens"]["platform_signal"] = torch.zeros(n)
    return runner


# ---- Public factory -------------------------------------------------------


def make_covid_env(
    *,
    init_seed: int = 0,
    feature_provider: Optional[Callable[[Runner], Tensor]] = None,
    signal_writer: Optional[Callable[[Runner, Tensor], None]] = None,
    state_extractor: Optional[Callable[[Runner], dict[str, Tensor]]] = None,
    keep_trajectory: bool = False,
    strict_signal: bool = True,
    initial_infections_fraction: float | None = None,
) -> AgentTorchEnvironment:
    """Construct an `AgentTorchEnvironment` driving the bundled covid sim.

    All four callables have sensible covid-flavored defaults; override any
    of them to change the feature space, the signal-injection format, or
    the supervised target shape.

    initial_infections_fraction: if not None, the runner_factory will seed
        this fraction of agents as INFECTED (disease_stage=2) immediately
        after `runner.init()`. This survives `Simulator.run`'s `env.reset`
        (which rebuilds the runner via the factory), unlike calling
        `seed_initial_infections(env, ...)` after `make_covid_env`.
        Without seeded infections the population stays effectively
        all-Susceptible across the rollout and the LM's isolation policy
        has nothing to gate transmission against.
    """
    if initial_infections_fraction is not None:
        frac = float(initial_infections_fraction)

        def _factory(seed: int) -> Runner:
            runner = build_covid_runner(seed)
            seed_initial_infections(runner, fraction=frac, seed=seed)
            return runner
    else:
        _factory = build_covid_runner

    return AgentTorchEnvironment(
        runner_factory=_factory,
        feature_provider=feature_provider or default_feature_provider,
        signal_writer=signal_writer or default_signal_writer,
        state_extractor=state_extractor or default_state_extractor,
        signal_path=("agents", "citizens", "platform_signal"),
        keep_trajectory=keep_trajectory,
        strict_signal=strict_signal,
        init_seed=init_seed,
    )
