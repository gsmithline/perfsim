"""Faithful Perdomo strategic-loan scenario using GiveMeSomeCredit.

Public API:
    PerdomoLoanConfig: dataclass with reproduction parameters.
    build_dataset(config): real or synthetic Dataset per config.
    build_world(dataset, mu, ...): StrategicLinearWorld using Perdomo's convention.
    run(config): runs the scenario end-to-end and returns a History.
"""

from perfsim.scenarios.perdomo_loan.config import (
    PERDOMO_COMPETITION,
    PERDOMO_FEATURE_COLS,
    PERDOMO_FILE,
    PERDOMO_LABEL_COL,
    PerdomoLoanConfig,
    build_dataset,
    make_kaggle_dataset,
    make_synthetic_dataset,
)
from perfsim.scenarios.perdomo_loan.reproduction import run
from perfsim.scenarios.perdomo_loan.world import build_world

__all__ = [
    "PERDOMO_COMPETITION",
    "PERDOMO_FEATURE_COLS",
    "PERDOMO_FILE",
    "PERDOMO_LABEL_COL",
    "PerdomoLoanConfig",
    "build_dataset",
    "build_world",
    "make_kaggle_dataset",
    "make_synthetic_dataset",
    "run",
]
