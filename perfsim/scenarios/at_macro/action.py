"""Perfsim-controlled earning-decision action substep for macro_economics.

Replaces agent_torch.models.macro_economics.substeps.earning.action.
WorkConsumptionPropensity. The bundled action calls an OpenAI LLM directly
to produce (work_propensity, consumption_propensity) per agent — we don't
want that. Perfsim's LM-as-recommender model produces a per-agent signal
which the env injects via signal_writer; this substep just reads it.

Behavior: read the per-agent platform_signal that perfsim's signal_writer
deposited at `state["agents"]["consumers"]["platform_signal"]`, sigmoid
it to (0, 1), and emit it as consumption_propensity. `will_work` is set
from a fixed default work-propensity (configurable via
`simulation_metadata.default_work_propensity`, fallback 0.95).

If you want the LM to ALSO drive labor supply, write a 2-channel signal
(consumption, work) and split it here — the env's signal_writer
controls the format.

Pattern A2 (fixed anchor): this action only reads `platform_signal`, never
mutates it; the adapter's strict_signal allclose check passes.
"""

from __future__ import annotations

import torch

from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers.distributions import StraightThroughBernoulli


class PerfsimEarningDecision(SubstepAction):
    """Reads platform_signal, sigmoids to consumption_propensity, emits
    (will_work, consumption_propensity).

    Output shape and order match the bundled WorkConsumptionPropensity so
    the downstream UpdateAssets transition consumes the same fields.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        # Default work propensity (probability of agent working this round).
        # 0.95 means most agents work unless the LM-or-env says otherwise.
        # Override by setting `simulation_metadata.default_work_propensity`
        # in the YAML.
        self.default_work_propensity = float(
            self.config["simulation_metadata"].get("default_work_propensity", 0.95)
        )
        self.st_bernoulli = StraightThroughBernoulli.apply

    def forward(self, state, observation):
        signal = state["agents"]["consumers"]["platform_signal"]
        consumption_propensity = torch.sigmoid(signal).reshape(-1, 1).to(self.device)
        # Fixed work propensity → sample via StraightThroughBernoulli for
        # differentiability through the downstream substeps. Per-agent
        # tensor so the bundled UpdateAssets transition sees the right shape.
        work_propensity = torch.full(
            (self.num_agents, 1),
            self.default_work_propensity,
            dtype=torch.float32,
            device=self.device,
        )
        will_work = self.st_bernoulli(work_propensity)
        return {
            self.output_variables[0]: will_work,
            self.output_variables[1]: consumption_propensity,
        }
