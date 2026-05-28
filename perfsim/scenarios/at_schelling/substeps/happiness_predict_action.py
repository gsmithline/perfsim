"""Policy/transition pair that lifts platform_signal into p_pred."""

from __future__ import annotations

import re

import torch

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepAction, SubstepTransition


def _get(state, var: str):
    return get_by_path(state, re.split("/", var))


class HappinessPredictAction(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observation):
        sig = _get(state, self.input_variables["platform_signal"])
        if sig.ndim == 2 and sig.shape[-1] == 1:
            sig = sig.squeeze(-1)
        p = sig.clamp(0.0, 1.0).float()
        return {self.output_variables[0]: p}


class WritePPred(SubstepTransition):
    """Copies the action-emitted p_pred into agents/residents/p_pred."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        p = action["residents"]["p_pred"]
        return {"p_pred": p}
