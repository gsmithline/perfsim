"""SubstepTransition: compute H_i and sample move_decision."""

from __future__ import annotations

import re

import torch

from agent_torch.core.distributions import StraightThroughBernoulli
from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepTransition


def _get(state, var: str):
    return get_by_path(state, re.split("/", var))


_st_bernoulli = StraightThroughBernoulli.apply


class MoveDecision(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = self.config["simulation_metadata"]
        self.lam = float(meta["lambda_"])
        self.hardness = float(meta.get("move_hardness", 8.0))
        per_type = meta.get("baseline_threshold_per_type")
        if per_type is not None:
            self.H_0_per_type = torch.tensor(list(per_type), dtype=torch.float32)
        else:
            self.H_0_per_type = None
        self.H_0_scalar = float(meta["baseline_threshold"])

    def forward(self, state, action):
        iv = self.input_variables
        p = _get(state, iv["p_pred"]).float()
        s = _get(state, iv["same_frac"]).float()
        types_ = _get(state, iv["type"]).long()

        if self.H_0_per_type is not None:
            H_0 = self.H_0_per_type.to(p.device)[types_]
        else:
            H_0 = torch.full_like(p, self.H_0_scalar)

        H_eff = (H_0 - self.lam * (p - 0.5)).clamp(0.0, 1.0)
        move_prob = torch.sigmoid(self.hardness * (H_eff - s))
        move = _st_bernoulli(move_prob)
        return {
            "effective_threshold": H_eff,
            "move_decision": move,
        }
