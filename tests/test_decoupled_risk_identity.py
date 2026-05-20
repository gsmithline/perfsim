"""Validation test 2 (gating): DPR(theta, theta) == PR(theta).

Definitional identity: when the deploy and eval models are the same, the
decoupled performative risk equals the performative risk.
"""

from __future__ import annotations

import torch

from perfsim.losses import BCEWithLogitsLoss, MSELoss
from perfsim.metrics import decoupled_risk, performative_risk
from perfsim.models import LinearModel
from perfsim.worlds import GaussianShiftWorld


def _make_world(d: int = 3, sigma: float = 0.01, batch: int = 256, seed: int = 0):
    A = 0.5 * torch.eye(d)
    b = torch.tensor([1.0, 0.5, -0.5])[:d].clone()
    w = GaussianShiftWorld(A=A, b=b, sigma_noise=sigma, batch_size=batch)
    w.reset(seed=seed)
    return w


def test_dpr_equals_pr_when_models_match() -> None:
    w = _make_world()
    m = LinearModel(in_features=3, out_features=1, bias=False)
    m.set_params(torch.tensor([0.7, -0.3, 0.2]))
    loss = MSELoss()
    pr = performative_risk(w, m, loss)
    dpr = decoupled_risk(w, m, m, loss)
    assert torch.allclose(pr, dpr, atol=1e-7)


def test_dpr_equals_pr_across_seeds() -> None:
    for seed in [0, 1, 42, 123]:
        w = _make_world(seed=seed)
        m = LinearModel(in_features=3, out_features=1, bias=False)
        m.set_params(torch.randn(3, generator=torch.Generator().manual_seed(seed)))
        loss = MSELoss()
        pr = performative_risk(w, m, loss)
        dpr = decoupled_risk(w, m, m, loss)
        assert torch.allclose(pr, dpr, atol=1e-7), f"failed at seed={seed}"


def test_dpr_equals_pr_for_alternate_loss() -> None:
    w = _make_world()
    m = LinearModel(in_features=3, out_features=1, bias=False)
    m.set_params(torch.tensor([0.1, 0.2, 0.3]))
    # GaussianShift produces continuous y but BCEWithLogits still computes a value
    # (treating y as a soft target); the DPR=PR identity is loss-agnostic.
    loss = BCEWithLogitsLoss()
    pr = performative_risk(w, m, loss)
    dpr = decoupled_risk(w, m, m, loss)
    assert torch.allclose(pr, dpr, atol=1e-7)
