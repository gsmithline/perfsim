"""Perdomo loan scenario configuration and dataset factories.

Defaults match the Perdomo et al. (ICML 2020) experimental setup as
closely as is practical with our abstractions:

- Class balancing: all positive cases + first N=10000 negative cases.
- mean / std standardization (sklearn `preprocessing.scale`).
- Strategic features restricted to columns (0, 5, 7) of the 10 features
  (`RevolvingUtilizationOfUnsecuredLines`, `NumberOfOpenCreditLinesAndLoans`,
  `NumberRealEstateLoansOrLines`).
- L2 regularization with bias excluded (`lam = 1/n` in Perdomo; default
  `weight_decay = 5e-5` here for the balanced n ~ 20k).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

from perfsim.core.dataset import Dataset
from perfsim.core.types import ConfigBase
from perfsim.datasets import InMemoryDataset, KaggleDataset

PERDOMO_FEATURE_COLS: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)
PERDOMO_LABEL_COL: str = "SeriousDlqin2yrs"
PERDOMO_COMPETITION: str = "GiveMeSomeCredit"
PERDOMO_FILE: str = "cs-training.csv"

# Perdomo notebook: `strat_features = np.array([1, 6, 8]) - 1` -> [0, 5, 7]
PERDOMO_STRAT_FEATURES: tuple[int, ...] = (0, 5, 7)


@dataclass(frozen=True)
class PerdomoLoanConfig(ConfigBase):
    """Config for the Perdomo loan reproduction."""

    mu: float = 1.0
    n_rounds: int = 30
    learner: str = "erm"
    learner_lr: float = 0.01
    learner_steps: int = 1
    weight_decay: float = 5e-5
    decay_bias: bool = False
    strat_features: Tuple[int, ...] = field(default_factory=lambda: PERDOMO_STRAT_FEATURES)
    balance_classes: bool = True
    balance_n_negatives: int = 10000
    standardize: bool = True
    robust: bool = False
    clip: float = 0.0
    seed: int = 0
    use_synthetic_fallback: bool = False
    synthetic_n: int = 1000
    synthetic_d: int = 10


def _balance_classes(
    raw: Dataset, *, n_negatives: int, seed: int
) -> InMemoryDataset:
    """All positive cases + first `n_negatives` negative cases, shuffled."""
    data = raw.load()
    x, y = data["x"], data["y"]
    pos_idx = (y == 1.0).nonzero(as_tuple=True)[0]
    neg_idx = (y == 0.0).nonzero(as_tuple=True)[0][:n_negatives]
    idx = torch.cat([pos_idx, neg_idx])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(idx.numel(), generator=g)
    idx = idx[perm]
    return InMemoryDataset({"x": x[idx], "y": y[idx]})


def make_kaggle_dataset(
    *,
    balance: bool = True,
    n_negatives: int = 10000,
    seed: int = 0,
) -> Dataset:
    """Build the GiveMeSomeCredit Dataset with Perdomo's feature / label columns.

    When `balance=True` (default), takes all positive rows plus the first
    `n_negatives` negative rows, shuffled deterministically. This matches
    Perdomo's `data_prep.load_data` setup.
    """
    raw: Dataset = KaggleDataset(
        competition=PERDOMO_COMPETITION,
        file=PERDOMO_FILE,
        label_col=PERDOMO_LABEL_COL,
        feature_cols=PERDOMO_FEATURE_COLS,
    )
    if not balance:
        return raw
    return _balance_classes(raw, n_negatives=n_negatives, seed=seed)


def make_synthetic_dataset(
    *,
    n: int = 1000,
    d: int = 10,
    seed: int = 0,
) -> InMemoryDataset:
    """Synthetic GMSC-like dataset for testing without Kaggle credentials.

    Features are standard normal; labels are drawn from a Bernoulli with
    logit linear in features. Not the replication claim.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    true_w = torch.randn(d, generator=g) / (d**0.5)
    logits = x @ true_w
    y = (torch.rand(n, generator=g) < torch.sigmoid(logits)).float()
    return InMemoryDataset({"x": x, "y": y})


def build_dataset(config: PerdomoLoanConfig) -> Dataset:
    """Pick the dataset implied by the config (real or synthetic)."""
    if config.use_synthetic_fallback:
        return make_synthetic_dataset(
            n=config.synthetic_n, d=config.synthetic_d, seed=config.seed
        )
    return make_kaggle_dataset(
        balance=config.balance_classes,
        n_negatives=config.balance_n_negatives,
        seed=config.seed,
    )
