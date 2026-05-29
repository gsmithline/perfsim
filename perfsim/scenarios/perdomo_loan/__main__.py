"""CLI entry point: python -m perfsim.scenarios.perdomo_loan ...

Examples:
    python -m perfsim.scenarios.perdomo_loan --synthetic --mu 1.0
    python -m perfsim.scenarios.perdomo_loan --mu 100 --n-rounds 25 --learner erm
    python -m perfsim.scenarios.perdomo_loan --mu 1000 --learner gradient --learner-lr 0.1
"""

from __future__ import annotations

import argparse

from perfsim.scenarios.perdomo_loan.config import (
    PERDOMO_STRAT_FEATURES,
    PerdomoLoanConfig,
)
from perfsim.scenarios.perdomo_loan.reproduction import run


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perfsim.scenarios.perdomo_loan",
        description="Reproduce Perdomo et al. (ICML 2020) strategic loan example.",
    )
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--learner", choices=("erm", "gradient"), default="erm")
    parser.add_argument("--learner-lr", type=float, default=0.01)
    parser.add_argument("--learner-steps", type=int, default=1)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="L2 coefficient; default None uses Juan's exact lam = 1/n.",
    )
    parser.add_argument(
        "--decay-bias",
        action="store_true",
        help="Include bias in L2 regularization. Perdomo excludes it (default off).",
    )
    parser.add_argument(
        "--strat-features",
        type=int,
        nargs="+",
        default=list(PERDOMO_STRAT_FEATURES),
        help="Feature column indices that can be strategically manipulated.",
    )
    parser.add_argument(
        "--no-balance",
        dest="balance_classes",
        action="store_false",
        help="Use the full unbalanced GMSC instead of Perdomo's pos + 10k-neg subset.",
    )
    parser.add_argument(
        "--balance-n-negatives",
        type=int,
        default=10000,
        help="Number of negative cases to include when balancing (Perdomo: 10000).",
    )
    parser.add_argument(
        "--no-standardize", dest="standardize", action="store_false"
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use median/IQR standardization (default off; Perdomo uses mean/std).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.0,
        help="Clip post-standardized features to [-clip, clip]. 0 disables (default).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synthetic",
        dest="use_synthetic_fallback",
        action="store_true",
        help="Use synthetic GMSC-like data; not the replication claim.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    config = PerdomoLoanConfig(
        mu=args.mu,
        n_rounds=args.n_rounds,
        learner=args.learner,
        learner_lr=args.learner_lr,
        learner_steps=args.learner_steps,
        weight_decay=args.weight_decay,
        decay_bias=args.decay_bias,
        strat_features=tuple(args.strat_features),
        balance_classes=args.balance_classes,
        balance_n_negatives=args.balance_n_negatives,
        standardize=args.standardize,
        robust=args.robust,
        clip=args.clip,
        seed=args.seed,
        use_synthetic_fallback=args.use_synthetic_fallback,
    )
    history = run(config)
    print(
        f"# Perdomo loan: mu={config.mu}, learner={config.learner}, "
        f"rounds={config.n_rounds}, balance={config.balance_classes}, "
        f"strat_features={list(config.strat_features)}, "
        f"wd={config.weight_decay}, synthetic={config.use_synthetic_fallback}"
    )
    print(f"# config hash: {config.content_hash()}")
    print(f"# rounds recorded: {len(history)}")
    for r in history.records:
        pr = r.get("PR")
        stab = r.get("stability_gap")
        pr_s = f"{pr.item():.6f}" if pr is not None else "n/a"
        stab_s = f"{stab.item():.6f}" if stab is not None else "n/a"
        print(f"  round={r['round']:3d}  PR={pr_s}  stability_gap={stab_s}")


if __name__ == "__main__":
    main()
