"""Build an `AgentTorchEnvironment` wired to AT's bundled macro_economics model.

Public entry point: `make_macro_env(seed=0, **overrides) -> AgentTorchEnvironment`.

Mirrors the structure of perfsim.scenarios.at_covid.env. Differences:
  - Registers `PerfsimEarningDecision` in place of the bundled
    `WorkConsumptionPropensity` LLM-calling action.
  - Registers the bundled transition substeps (UpdateAssets,
    UpdateAssetsGoods, UpdateMacroRates, UpdateFinancialMarket,
    WriteActionToState) — these import from `AgentTorch.*` (capital A),
    which the `install_agenttorch_alias` shim makes resolvable.
  - Default feature_provider uses agent `age` (1-D feature).
  - Default state_extractor returns (x=age, y=assets, agent_idx). The y
    field is what the LM is supervised on — for the at_macro experiment
    we expect the user to override it with a target derived from current
    macro state (inflation, unemployment) and per-agent assets, the
    economic analog of the at_covid exposure-aware target.

Honest TODOs flagged in comments below — the macro model has more moving
parts than covid and the user will want to tune them.
"""

from __future__ import annotations

import tempfile
from typing import Callable, Optional

import torch
from torch import Tensor

from perfsim.scenarios.at_macro._compat import (
    bundled_nyc_dir,
    bundled_macro_yaml,
    install_agenttorch_alias,
    patched_macro_yaml,
    should_register_resolvers,
    subsample_population_dir,
)


# Install the AgentTorch (capital-A) alias before importing the bundled
# macro_economics transition substeps; they reference AgentTorch.* at
# module-import time.
install_agenttorch_alias()

from agent_torch.core import Registry, Runner  # noqa: E402
from agent_torch.core.helpers import read_config  # noqa: E402
from agent_torch.models.macro_economics.substeps.earning.transition import (  # noqa: E402
    UpdateAssets,
    WriteActionToState,
)
from agent_torch.models.macro_economics.substeps.consumption.transition import (  # noqa: E402
    UpdateAssetsGoods,
)
from agent_torch.core.substep import SubstepTransition  # noqa: E402


class _PatchedMacroRates(SubstepTransition):
    """Patched UpdateMacroRates that fixes the shape mismatch in the bundled code.

    The bundled UpdateMacroRates does:
        matmul(one_hot(t, num_timesteps), self.external_UAC)
    expecting UAC shape (num_timesteps, 3). But the YAML defines it as
    shape [1] with value 0.7. We expand the scalar to (num_timesteps, 3)
    by broadcasting the single value as the first coefficient and zeroing
    the rest. This recovers a simplified version of the unemployment
    equation: unemp = UAC * log(labor_force).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.max_rate_change = self.config["simulation_metadata"][
            "maximum_rate_of_change_of_wage"
        ]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        uac_raw = self.learnable_args.get("unemployment_adaptation_coefficient")
        if uac_raw is not None:
            uac_val = torch.tensor(uac_raw, dtype=torch.float32, requires_grad=True)
        else:
            uac_val = torch.tensor([0.7], dtype=torch.float32, requires_grad=True)

        if uac_val.numel() < self.num_timesteps * 3:
            expanded = torch.zeros(self.num_timesteps, 3)
            expanded[:, 0] = uac_val.flatten()[0]
            self.external_UAC = torch.nn.Parameter(expanded)
        else:
            self.external_UAC = torch.nn.Parameter(
                uac_val.reshape(self.num_timesteps, 3)
            )

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        import torch.nn.functional as F
        timestep_tensor = torch.tensor([timestep])
        return F.one_hot(timestep_tensor, num_classes=num_timesteps).to(self.device)

    def forward(self, state, action):
        import re
        from agent_torch.core.helpers import get_by_path

        t = int(state["current_step"])
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        working_status = get_by_path(
            state, re.split("/", self.input_variables["will_work"])
        )
        imbalance = get_by_path(
            state, re.split("/", self.input_variables["imbalance"])
        )
        hourly_wage = get_by_path(
            state, re.split("/", self.input_variables["hourly_wage"])
        )
        unemployment_rate = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate"])
        )
        labor_force = get_by_path(
            state, re.split("/", self.input_variables["labor_force"])
        )

        uac = torch.matmul(
            time_step_one_hot.float().unsqueeze(dim=0), self.external_UAC
        ).squeeze([0, 1])

        total_labor_force = torch.sum(working_status)

        current_unemp = (
            uac[0] * torch.log(total_labor_force.clamp(min=1.0))
            + uac[2]
        )

        unemployment_rate = unemployment_rate + (
            current_unemp * time_step_one_hot
        )
        labor_force = labor_force + (total_labor_force * time_step_one_hot)

        omega = imbalance.float()
        r1, r2 = self.max_rate_change * omega, torch.tensor(0.0)
        sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
        new_hourly_wages = hourly_wage + hourly_wage * sampled_omega

        return {
            self.output_variables[0]: new_hourly_wages,
            self.output_variables[1]: unemployment_rate,
            self.output_variables[2]: labor_force,
        }


class _PatchedFinancialMarket(SubstepTransition):
    """Wires through the bundled UpdateFinancialMarket logic directly.

    The bundled code imports from `AgentTorch` (capital A) which the
    _compat shim handles. The logic itself is sound — Taylor-rule interest
    rate, price adjustment from imbalance, inflation from price change.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        import re
        from agent_torch.core.helpers import get_by_path

        number_of_months = state["current_step"] + 1

        inflation_rate = get_by_path(
            state, re.split("/", self.input_variables["inflation_rate"])
        )
        unemployment_rate = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate"])
        )
        price_of_goods = get_by_path(
            state, re.split("/", self.input_variables["price_of_goods"])
        )
        cumulative_price_of_goods = get_by_path(
            state, re.split("/", self.input_variables["cumulative_price_of_goods"])
        )
        imbalance = get_by_path(
            state, re.split("/", self.input_variables["imbalance"])
        )

        # Taylor rule interest rate
        rn = self.config["simulation_metadata"]["natural_interest_rate"]
        un = self.config["simulation_metadata"]["natural_unemployment_rate"]
        pit = self.config["simulation_metadata"]["target_inflation_rate"]
        a_i = self.config["simulation_metadata"]["inflation_adaptation_coefficient"]
        a_u = self.config["simulation_metadata"]["unemployment_adaptation_macro"]

        new_interest_rate = rn + pit + a_i * (inflation_rate - pit) + a_u * (un - unemployment_rate)
        new_interest_rate = torch.max(new_interest_rate, torch.zeros_like(new_interest_rate))

        # Price adjustment from imbalance
        omega = imbalance.float()
        max_rate_change = self.config["simulation_metadata"]["maximum_rate_of_change_of_price"]
        if omega > 0:
            r2 = max_rate_change * omega
            sampled_omega = -r2 * torch.rand(1, 1)
        else:
            r1 = max_rate_change * omega
            sampled_omega = r1 * torch.rand(1, 1)
        new_price_of_goods = price_of_goods * (1 + sampled_omega)

        new_cumulative = cumulative_price_of_goods + new_price_of_goods.squeeze()
        avg_price = new_cumulative / number_of_months
        new_inflation_rate = (price_of_goods - avg_price) / avg_price.clamp(min=1e-8)

        return {
            self.output_variables[0]: new_interest_rate,
            self.output_variables[1]: new_price_of_goods,
            self.output_variables[2]: new_cumulative,
            self.output_variables[3]: new_inflation_rate,
        }
from agent_torch.models.macro_economics.substeps.utils import (  # noqa: E402
    get_population_size,
    initialize_id,
    load_population_attribute,
    random_normal_col_by_col,
)
from agent_torch.core.helpers.environment import grid_network  # noqa: E402

from perfsim.adapters.agenttorch import AgentTorchEnvironment  # noqa: E402
from perfsim.scenarios.at_macro.action import PerfsimEarningDecision  # noqa: E402


# ---- Default callables ----------------------------------------------------


def default_feature_provider(runner: Runner) -> Tensor:
    """Per-agent age as a (N, 1) float feature."""
    return runner.state["agents"]["consumers"]["age"].float().detach().reshape(-1, 1)


def default_signal_writer(runner: Runner, preds: Tensor) -> None:
    """Squeeze predictions to (N,) and store at `agents/consumers/platform_signal`.

    Detaches before writing so that the non-grad `run` path is the safe
    default. For gradient measurement through `grad_run`, pass
    `default_signal_writer_grad` instead.
    """
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["consumers"]["platform_signal"] = preds.detach().clone()


def default_signal_writer_grad(runner: Runner, preds: Tensor) -> None:
    """Non-detaching variant for use with `grad_run`."""
    if preds.ndim == 2 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    runner.state["agents"]["consumers"]["platform_signal"] = preds.clone()


def default_state_extractor(runner: Runner) -> dict[str, Tensor]:
    """Return `(x=age, y=current_assets, agent_idx)` as perfsim supervised Data.

    Assets is (N, num_timesteps) — each column holds the value at that
    timestep. We extract the current step's column as y. Falls back to
    the last non-zero column if current_step overflows.
    """
    consumers = runner.state["agents"]["consumers"]
    age = consumers["age"].float().detach().reshape(-1, 1)
    assets_all = consumers["assets"].float().detach()
    t = int(runner.state.get("current_step", 0))
    if assets_all.ndim == 2:
        col = min(t, assets_all.shape[1] - 1)
        assets = assets_all[:, col].reshape(-1, 1)
    else:
        assets = assets_all.reshape(-1, 1)
    n = age.shape[0]
    return {
        "x": age,
        "y": assets,
        "agent_idx": torch.arange(n),
    }


def _build_registry() -> Registry:
    """Register perfsim's action + bundled transitions + initializers.

    Names must match the `generator:` fields in the bundled YAML
    (config_100_agents.yaml et al). Our PerfsimEarningDecision is
    registered under "WorkConsumptionPropensity" so the YAML's policy
    block picks it up in place of the bundled LLM-calling action.
    """
    reg = Registry()
    # AT registers helpers by the YAML's dict-key name, not the generator
    # class name. The macro YAML names the policy "get_work_consumption_decision"
    # and the transitions "update_assets", "write_action_to_state",
    # "update_assets_and_goods", "update_macro_rates",
    # "update_financial_market" — use those names.
    reg.register(PerfsimEarningDecision, "get_work_consumption_decision", key="policy")
    reg.register(UpdateAssets, "update_assets", key="transition")
    reg.register(WriteActionToState, "write_action_to_state", key="transition")
    reg.register(UpdateAssetsGoods, "update_assets_and_goods", key="transition")
    # Patched versions — fix shape mismatch in bundled UpdateMacroRates,
    # re-implement UpdateFinancialMarket without the capital-A import issue.
    reg.register(_PatchedMacroRates, "update_macro_rates", key="transition")
    reg.register(_PatchedFinancialMarket, "update_financial_market", key="transition")
    # Initializers — the bundled macro YAML references these by these names.
    reg.register(load_population_attribute, "load_population_attribute", key="initialization")
    reg.register(initialize_id, "initialize_id", key="initialization")
    reg.register(get_population_size, "get_population_size", key="initialization")
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")
    # Network helper for the consumers/opinion_network in the YAML.
    reg.register(grid_network, "grid", key="network")
    return reg


def build_macro_runner(
    seed: int = 0,
    *,
    yaml_name: str = "config_100_agents.yaml",
    n_agents: int = 100,
    population_dir: str | None = None,
) -> Runner:
    """Construct a fully-initialized agent_torch.Runner for the bundled
    macro_economics model.

    yaml_name: which bundled YAML to use. Default is the smallest
        (100-agent) config. Other options:
          - "config.yaml" (full, large; needs scaled-down env vars)
          - "config_nyc_100_agents.yaml" (NYC-specific small)
    n_agents: subsample size. Bundled NYC pickles have ~2.7M rows but the
        YAML's `num_agents` is 100, causing shape mismatches without
        subsampling. Pass the same value as the YAML's num_agents.
    population_dir: optional override for the population path. If None,
        a fresh temp dir with the bundled NYC data subsampled to
        `n_agents` is created.
    """
    import pathlib

    torch.manual_seed(int(seed))
    src_yaml = bundled_macro_yaml(name=yaml_name)
    if population_dir is not None:
        pop_dir = pathlib.Path(population_dir)
    elif yaml_name == "config.yaml":
        pop_dir = bundled_nyc_dir()
    else:
        pop_dir = subsample_population_dir(n_agents=n_agents)
    yaml_path = patched_macro_yaml(src_yaml=src_yaml, population_dir=pop_dir)
    config = read_config(yaml_path, register_resolvers=should_register_resolvers())
    reg = _build_registry()
    runner = Runner(config, reg)
    runner.init()

    n = config["simulation_metadata"]["num_agents"]
    runner.state["agents"]["consumers"]["platform_signal"] = torch.zeros(n)
    return runner


# ---- Public factory -------------------------------------------------------


def make_macro_env(
    *,
    init_seed: int = 0,
    yaml_name: str = "config_100_agents.yaml",
    n_agents: int = 100,
    population_dir: str | None = None,
    feature_provider: Optional[Callable[[Runner], Tensor]] = None,
    signal_writer: Optional[Callable[[Runner, Tensor], None]] = None,
    state_extractor: Optional[Callable[[Runner], dict[str, Tensor]]] = None,
    keep_trajectory: bool = False,
    strict_signal: bool = True,
) -> AgentTorchEnvironment:
    """Construct an AgentTorchEnvironment driving the bundled macro_economics
    sim.

    All callables have sensible defaults; override any of them to change
    feature space, signal-injection format, or supervised target shape.

    keep_trajectory: True → AT runner state persists across env.run calls
        (round t+1 starts from end of round t). Almost always what you
        want for a performative loop; mirror of covid.
    """
    def _factory(seed: int) -> Runner:
        return build_macro_runner(
            seed,
            yaml_name=yaml_name,
            n_agents=n_agents,
            population_dir=population_dir,
        )

    return AgentTorchEnvironment(
        runner_factory=_factory,
        feature_provider=feature_provider or default_feature_provider,
        signal_writer=signal_writer or default_signal_writer,
        state_extractor=state_extractor or default_state_extractor,
        signal_path=("agents", "consumers", "platform_signal"),
        keep_trajectory=keep_trajectory,
        strict_signal=strict_signal,
        init_seed=init_seed,
    )
