"""Outcome probes for the AI-mediated loop.

These measure the things that actually decide whether mediation is harmful, as
opposed to merely cosmetic:

- recoverability(data): best-achievable AUC of a strong probe predicting y from
  the (mediated) features. This is the operational measure of information about
  the label. A DROP across rounds = consequential information loss; FLAT =
  cosmetic homogenization only (signal survives).
- style_variance(data, style_mask): total variance in the style columns. A drop
  is descriptive homogenization, NOT sufficient evidence of harm on its own.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perfsim.core.types import Data


def recoverability(
    data: Data,
    *,
    seed: int = 0,
    test_frac: float = 0.3,
    max_iter: int = 1000,
) -> float:
    """Best-achievable test AUC predicting y from x (a strong linear probe).

    Returns 0.5 when the label is degenerate (single class) on either split,
    i.e. no information is recoverable.
    """
    x = data["x"].detach().cpu().numpy()
    y = data["y"].detach().cpu().reshape(-1).numpy()
    y = (y > 0.5).astype(int)
    if np.unique(y).size < 2:
        return 0.5
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_frac, random_state=seed, stratify=y
    )
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return 0.5
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(scaler.transform(x_tr), y_tr)
    p = clf.predict_proba(scaler.transform(x_te))[:, 1]
    return float(roc_auc_score(y_te, p))


def style_variance(data: Data, style_mask: torch.Tensor) -> float:
    """Total variance summed over the style (z) columns. Descriptive only."""
    x = data["x"].detach().cpu()
    style = x[:, style_mask]
    return float(style.var(dim=0, unbiased=False).sum().item())


def model_auc(model, data: Data) -> float:
    """Test AUC of the deployed platform model on `data` (e.g. raw held-out).

    Unlike `recoverability` (best-achievable info in the data), this measures how
    well the CURRENT theta predicts y, so it reflects the retrain regime. Returns
    0.5 for a degenerate label.
    """
    y = data["y"].detach().cpu().reshape(-1).numpy()
    y = (y > 0.5).astype(int)
    if np.unique(y).size < 2:
        return 0.5
    with torch.no_grad():
        out = model(data["x"]).detach().cpu().reshape(-1)
    return float(roc_auc_score(y, out.numpy()))
