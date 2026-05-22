"""Perfsim-controlled isolation-decision action substep for covid.

Replaces agent_torch.models.covid.substeps.new_transmission.action.MakeIsolationDecision.
The bundled action is unloadable in agent_torch 0.6.0 because it imports
langchain 0.x APIs; the langchain shim in `_compat.py` is enough to *import*
the module but the bundled action's behavior depends on LLM machinery we do
not want.

Behavior: read the per-agent platform signal that perfsim's `signal_writer`
deposited at `state["agents"]["citizens"]["platform_signal"]`, sigmoid it to
(0, 1), and return it as `will_isolate`. The bundled NewTransmission
transition substep consumes `will_isolate` to gate disease spread per agent.

Pattern: this action only reads `platform_signal`. It does not mutate it,
so the adapter's strict_signal allclose check passes (A2 contract).
"""

from __future__ import annotations

import torch

from agent_torch.core.substep import SubstepAction


class PerfsimIsolationDecision(SubstepAction):
    """Reads platform_signal, sigmoids, returns as will_isolate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])

    def forward(self, state, observation):
        signal = state["agents"]["citizens"]["platform_signal"]
        will_isolate = torch.sigmoid(signal).reshape(-1, 1).to(self.device)
        return {self.output_variables[0]: will_isolate}
